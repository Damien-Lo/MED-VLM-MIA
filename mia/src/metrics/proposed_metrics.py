import numpy as np
from collections import defaultdict
import torch
from src.data.augmentations import get_augmentations
import sys
import ast

"""
Our proposed metric computation functions are listed here

Metric functions design structure

1. Input:

A value of meta_metrics dictionary has the structure as follows.

{
    "orig": [
        [
            (metric of a sample 0 in the batch),
            (metric of a sample 1 in the batch), ...
        ]
    ],
    "aug_1: [
        [
            (metric of a sample 0 in the batch, processed with a setting of aug_1),
            (metric of a sample 1 in the batch, processed with another setting of aug_1), ...
        ],
        [
            (metric of a sample 0 in the batch, processed with a setting of aug_1),
            (metric of a sample 1 in the batch, processed with another setting of aug_1), ...
        ]
    ]
}

2. Output:

Computed img_metrics that has following structure

If the metric has different sub_settings (such as different ratios)

{
    "Metric_1" : [ # ex: Min_30%_entropy 
        finalized score from the sample 0 in the batch,
        finalized score from the sample 1 in the batch, ...
    ],
    "Metric_2" : [
        finalized score from the sample 0 in the batch,
        finalized score from the sample 1 in the batch, ...
    ],
    ...
}

If the metric does not have such settings:

[
    finalized score from the sample 0 in the batch,
    finalized score from the sample 1 in the batch, ...
]

This will be automatically aggregated afterwards for AUC.

"""




def cross_entropy_mink(per_token_ce, cfg):
    ratio = cfg.ratio

    aug_names = list(per_token_ce.keys())
    result = dict()
    
    for _sample_idx, per_token_loss in enumerate(per_token_ce["orig"][0]):
        _loss_diff = dict()
        for aug_name in aug_names:
            if aug_name == "orig":
                continue
            current_augs = per_token_ce[aug_name]
            for _aug in current_augs:
                for _ratio in ratio:
                    k_length = int(len(_aug[_sample_idx])*_ratio)
                    if k_length == 0:
                        k_length = 1
                    if _ratio not in _loss_diff:
                        _loss_diff[_ratio] = list()

                    # Convert to numpy array if needed and get indices
                    aug_values = np.array(_aug[_sample_idx])
                    aug_max_idx = np.argsort(aug_values)[-k_length:]
                    aug_max_avg = aug_values[aug_max_idx].mean()
                    
                    # Convert per_token_loss to numpy array if needed
                    orig_values = np.array(per_token_loss)
                    orig_max_avg = orig_values[aug_max_idx].mean()
                    
                    _loss_diff[_ratio].append(orig_max_avg - aug_max_avg)

        for _key in _loss_diff:
            result_key = f"Min_{_key*100}% Cross_Entro_Augs"
            if result_key not in result: 
                result[result_key] = list()
            result[result_key].append(np.mean(_loss_diff[_key]).item())

    return result


def cross_entropy_diff_mink(per_token_ce, cfg):
    ratio = cfg.ratio

    aug_names = list(per_token_ce.keys())
    result = dict()
    
    for _sample_idx, per_token_loss in enumerate(per_token_ce["orig"][0]):
        _loss_diff = dict()
        for aug_name in aug_names:
            if aug_name == "orig":
                continue
            current_augs = per_token_ce[aug_name]
            for _aug in current_augs:
                # Convert to numpy arrays
                orig_values = np.array(per_token_loss)
                aug_values = np.array(_aug[_sample_idx])
                _diff = orig_values - aug_values
                
                for _ratio in ratio:
                    k_length = int(len(_diff)*_ratio)
                    if k_length == 0:
                        k_length = 1
                    if _ratio not in _loss_diff:
                        _loss_diff[_ratio] = list()
                    
                    _diff_max = np.sort(_diff[k_length:]) # Bigger difference is bigger in negative
                    _loss_diff[_ratio].append(np.mean(_diff_max))

        for _key in _loss_diff:
            result_key = f"Min_{_key*100}% Cross_Entro_Augs"
            if result_key not in result: 
                result[result_key] = list()
            result[result_key].append(-1 * np.mean(_loss_diff[_key]).item()) # members has smaller score flip

    return result

def renyi_kl_div_maxk(renyi_probs, metric_cfg, cfg, eps=1e-12):
    result = dict()
    meta = dict()
    
    ratio = metric_cfg.ratio
    _, aug_desc_dict = get_augmentations(cfg)
    
    original_probs = renyi_probs['orig'][0]
    number_of_samples = len(original_probs)
    
    setting_version_accumilator = metric_cfg.augmentation_accumilator
    aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

    # Type: {max: [], avg: []}[augs]
    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in renyi_probs.items():
        if aug == "orig":
            continue
        
        all_settings_in_aug = dict()                                   #Shape: {setting: [sample, kld]}

        # Processed Data
        for setting_idx, setting in enumerate(settings):
            key = aug_desc_dict[aug][setting_idx]
            all_samples_in_setting_values = list()          # Shape: [samples, kld]
            for sample_idx, aug_probs in enumerate(setting):
                org = torch.stack(original_probs[sample_idx]).float().cpu().numpy()
                org_log = np.log(org + eps)
                aug_log = np.log(torch.stack(aug_probs).float().cpu().numpy() + eps)

                kl = kl_div_per_token(org, org_log, aug_log) # KL: 1D vector
                
                # Append Values to Respective Data Storage
                all_samples_in_setting_values.append(kl)
                
            all_settings_in_aug[str(key)] = all_samples_in_setting_values
            # all_settings_in_aug {setting: [sample, kld(1D)]}

        # For Aug, Push sampled raw kl_array to meta return
        meta[aug] = all_settings_in_aug
        
        
        if aug not in result:
            result[aug] = dict()
        # Get Scores for Rawest kld values setting by setting no aggrigation
        if 'none' in setting_version_accumilator:
            for _ratio in ratio:
                for setting_name, setting_values in all_settings_in_aug.items():
                    # key = f"Max_{_ratio}_{cfg.suffix}_no_agg_aug_{aug}_setting_{setting_idx}"
                    key = ast.literal_eval(setting_name)
                    key['k_ratio'] = _ratio
                                        
                    sample_scores = list()
                    # Cant use np.array because for some reason some samples have different sequence lengths, but even enventually for when using different description lengths
                    # just keep it generalisabole to python lists
                    for sample in setting_values:
                        k_length = max(1, int(_ratio * len(sample)))
                        # Traditional divergence-MIA convention: higher top-k KLD/Renyi divergence -> higher score -> predicted member (no flip).
                        sample_scores.append((np.mean(np.sort(sample)[-k_length:])).item()) 
                        
                    result[aug][str(key)] = sample_scores
                    
        
        
        # AGGIGATION
        # For each sample, aggrigate across the settings
        #TODO: for some reason, all_settings_in_aug is inhomogeneous, so can't np.array need to check whether that is an issue
        # or need to adapt structure
        setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
        if 'max' in setting_version_accumilator:
            if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['max'] = list()
            setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(list(all_settings_in_aug.values())), axis=0))
        if 'avg' in setting_version_accumilator:
            if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
            setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(list(all_settings_in_aug.values())), axis=0))
        aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)
        
             
    # For All Augs
    # For each sample, aggrigate across the augmentation
    final_combinations = dict()             # Shape: {combination: [sample, kld]}
    if 'max' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.max(np.array(stacked_kl_divs),axis=0).tolist()
    
    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.mean(np.array(stacked_kl_divs),axis=0).tolist()
    
    if len(final_combinations) != 0:
        result['aggregated'] = dict()
        
    # Min-k
    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{metric_cfg.suffix}_kld_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                # Traditional divergence-MIA convention: higher top-k KLD/Renyi divergence -> higher score -> predicted member (no flip).
                sample_scores.append(float(np.mean(np.sort(sample)[-k_length:]))) 
            result['aggregated'][key] = sample_scores 
    return result, meta


def kl_div_per_token(org_probs, org_log_probs, aug_log_probs):
    return np.sum(org_probs * (org_log_probs - aug_log_probs),axis=1)
    





def renyi_divergence_maxk(probs, metric_cfg, cfg, eps=1e-12):
    # print("Renyi-Div Metric")
    alpha = metric_cfg.alpha
    result = dict()
    meta = dict()
    
    ratio = metric_cfg.ratio
    _, aug_desc_dict = get_augmentations(cfg)
    
    original_probs = probs['orig'][0]
    number_of_samples = len(original_probs)
    
    setting_version_accumilator = metric_cfg.augmentation_accumilator
    aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

    # Type: {max: [], avg: []}[augs]
    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in probs.items():
        if aug == "orig":
            continue
        
        all_settings_in_aug = dict()                                     #Shape: [setting, sample, kld]

         # Processed Data
        for setting_idx, setting in enumerate(settings):
            key = aug_desc_dict[aug][setting_idx]
            all_samples_in_setting_values = list()          # Shape: [samples, kld]
            for sample_idx, aug_probs in enumerate(setting):
                org = original_probs[sample_idx]
                
                kl = renyi_div_per_token(org, aug_probs, alpha, eps)
                
                # Append Values to Respective Data Storage
                all_samples_in_setting_values.append(kl)
                
            all_settings_in_aug[str(key)] = all_samples_in_setting_values
            # all_settings_in_aug {setting: [sample, kld(1D)]}

        # For Aug, Push sampled raw kl_array to meta return
        meta[aug] = all_settings_in_aug
        
        
        if aug not in result:
            result[aug] = dict()
        # Get Scores for Rawest kld values setting by setting no aggrigation
        if 'none' in setting_version_accumilator:
            for _ratio in ratio:
                for setting_name, setting_values in all_settings_in_aug.items():
                    # key = f"Max_{_ratio}_{cfg.suffix}_no_agg_aug_{aug}_setting_{setting_idx}"
                    key = ast.literal_eval(setting_name)
                    key['k_ratio'] = _ratio
                                        
                    sample_scores = list()
                    # Cant use np.array because for some reason some samples have different sequence lengths, but even enventually for when using different description lengths
                    # just keep it generalisabole to python lists
                    for sample in setting_values:
                        k_length = max(1, int(_ratio * len(sample)))
                        # Traditional divergence-MIA convention: higher top-k KLD/Renyi divergence -> higher score -> predicted member (no flip).
                        sample_scores.append((np.mean(np.sort(sample)[-k_length:])).item()) 
                        
                    result[aug][str(key)] = sample_scores
        
        # AGGIGATION
        # For each sample, aggrigate across the settings
        #TODO: for some reason, all_settings_in_aug is inhomogeneous, so can't np.array need to check whether that is an issue
        # or need to adapt structure
        setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
        if 'max' in setting_version_accumilator:
            if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['max'] = list()
            setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(list(all_settings_in_aug.values())), axis=0))
        if 'avg' in setting_version_accumilator:
            if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
            setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(list(all_settings_in_aug.values())), axis=0))
        aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)
        
             
    # For All Augs
    # For each sample, aggrigate across the augmentation
    final_combinations = dict()             # Shape: {combination: [sample, kld]}
    if 'max' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.max(np.array(stacked_kl_divs),axis=0).tolist()
    
    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.mean(np.array(stacked_kl_divs),axis=0).tolist()
    
    if len(final_combinations) != 0:
        result['aggregated'] = dict()
        
    # Min-k
    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{metric_cfg.suffix}_kld_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                # Traditional divergence-MIA convention: higher top-k KLD/Renyi divergence -> higher score -> predicted member (no flip).
                sample_scores.append(float(np.mean(np.sort(sample)[-k_length:]))) 
            result['aggregated'][key] = sample_scores 
    return result, meta


def renyi_div_per_token(org_probs, aug_probs, alpha, eps=1e-12):
    org_probs = np.clip(org_probs, eps, 1.0)
    aug_probs = np.clip(aug_probs, eps, 1.0)
    divergence = (1 / (alpha - 1)) * np.log(np.sum(org_probs**alpha * aug_probs**(1 - alpha), axis=1) + eps)
    return divergence










def renyi_kl_div_ripple_maxk(renyi_probs, metric_cfg, cfg, eps=1e-12):
    result = dict()
    meta = dict()
    
    ratio = metric_cfg.ratio
    _, aug_desc_dict = get_augmentations(cfg)
    
    original_probs = renyi_probs['orig'][0]
    number_of_samples = len(original_probs)
    
    setting_version_accumilator = metric_cfg.augmentation_accumilator
    aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

    # Type: {max: [], avg: []}[augs]
    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in renyi_probs.items():
        if aug == "orig":
            continue
        
        all_settings_in_aug = dict()                                   #Shape: {setting: [sample, kld]}

        # Processed Data
        for setting_idx, setting in enumerate(settings):
            key = aug_desc_dict[aug][setting_idx]
            all_samples_in_setting_values = list()          # Shape: [samples, kld]
            for sample_idx, aug_probs in enumerate(setting):
                org = torch.stack(original_probs[sample_idx]).float().cpu().numpy()
                org_log = np.log(org + eps)
                aug_log = np.log(torch.stack(aug_probs).float().cpu().numpy() + eps)

                kl = kl_div_per_token(org, org_log, aug_log) # KL: 1D vector
                
                # Append Values to Respective Data Storage
                all_samples_in_setting_values.append(kl)
                
            all_settings_in_aug[str(key)] = all_samples_in_setting_values
            # all_settings_in_aug {setting: [sample, kld(1D)]}

        # For Aug, Push sampled raw kl_array to meta return
        meta[aug] = all_settings_in_aug
        
        
        if aug not in result:
            result[aug] = dict()
        # Get Scores for Rawest kld values setting by setting no aggrigation
        if "none" in setting_version_accumilator:
            for slice_idx in range(cfg.img_metrics.num_of_slices):
                for _ratio in ratio:
                    for setting_name, setting_values in all_settings_in_aug.items():
                        key = ast.literal_eval(setting_name)
                        key["slice_segment"] = (slice_idx, cfg.img_metrics.num_of_slices)
                        key["k_ratio"] = _ratio

                        sample_scores = list()
                        for sample in setting_values:
                            sliced = get_token_slice(sample, slice_idx, cfg.img_metrics.num_of_slices)
                            if sliced is None or len(sliced) == 0:
                                sample_scores.append(float("nan"))
                                continue

                            k_length = max(1, int(_ratio * len(sliced)))
                            sample_scores.append((np.mean(np.sort(sliced)[-k_length:])).item())

                        result[aug][str(key)] = sample_scores
                    
        
        
        # AGGIGATION
        # For each sample, aggrigate across the settings
        #TODO: for some reason, all_settings_in_aug is inhomogeneous, so can't np.array need to check whether that is an issue
        # or need to adapt structure
        setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
        if 'max' in setting_version_accumilator:
            if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['max'] = list()
            setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(list(all_settings_in_aug.values())), axis=0))
        if 'avg' in setting_version_accumilator:
            if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
            setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(list(all_settings_in_aug.values())), axis=0))
        aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)
        
             
    # For All Augs
    # For each sample, aggrigate across the augmentation
    final_combinations = dict()             # Shape: {combination: [sample, kld]}
    if 'max' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.max(np.array(stacked_kl_divs),axis=0).tolist()
    
    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.mean(np.array(stacked_kl_divs),axis=0).tolist()
    
    if len(final_combinations) != 0:
        result['aggregated'] = dict()
        
    # Min-k
    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{metric_cfg.suffix}_kld_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                # Traditional divergence-MIA convention: higher top-k KLD/Renyi divergence -> higher score -> predicted member (no flip).
                sample_scores.append(float(np.mean(np.sort(sample)[-k_length:]))) 
            result['aggregated'][key] = sample_scores 
    return result, meta










def renyi_divergence_ripple_maxk(probs, metric_cfg, cfg, eps=1e-12):
    # print("Renyi-Div Metric")
    alpha = metric_cfg.alpha
    result = dict()
    meta = dict()
    
    ratio = metric_cfg.ratio
    _, aug_desc_dict = get_augmentations(cfg)
    
    original_probs = probs['orig'][0]
    number_of_samples = len(original_probs)
    
    setting_version_accumilator = metric_cfg.augmentation_accumilator
    aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

    # Type: {max: [], avg: []}[augs]
    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in probs.items():
        if aug == "orig":
            continue
        
        all_settings_in_aug = dict()                                     #Shape: [setting, sample, kld]

         # Processed Data
        for setting_idx, setting in enumerate(settings):
            key = aug_desc_dict[aug][setting_idx]
            all_samples_in_setting_values = list()          # Shape: [samples, kld]
            for sample_idx, aug_probs in enumerate(setting):
                org = original_probs[sample_idx]
                
                kl = renyi_div_per_token(org, aug_probs, alpha, eps)
                
                # Append Values to Respective Data Storage
                all_samples_in_setting_values.append(kl)
                
            all_settings_in_aug[str(key)] = all_samples_in_setting_values
            # all_settings_in_aug {setting: [sample, kld(1D)]}

        # For Aug, Push sampled raw kl_array to meta return
        meta[aug] = all_settings_in_aug
        
        
        if aug not in result:
            result[aug] = dict()
        # Get Scores for Rawest kld values setting by setting no aggrigation
        if "none" in setting_version_accumilator:
            for slice_idx in range(cfg.img_metrics.num_of_slices):
                for _ratio in ratio:
                    for setting_name, setting_values in all_settings_in_aug.items():
                        key = ast.literal_eval(setting_name)
                        key["slice_segment"] = (slice_idx, cfg.img_metrics.num_of_slices)
                        key["k_ratio"] = _ratio

                        sample_scores = list()
                        for sample in setting_values:
                            sliced = get_token_slice(sample, slice_idx, cfg.img_metrics.num_of_slices)
                            if sliced is None or len(sliced) == 0:
                                sample_scores.append(float("nan"))
                                continue

                            k_length = max(1, int(_ratio * len(sliced)))
                            sample_scores.append((np.mean(np.sort(sliced)[-k_length:])).item())

                        result[aug][str(key)] = sample_scores
        
        # AGGIGATION
        # For each sample, aggrigate across the settings
        #TODO: for some reason, all_settings_in_aug is inhomogeneous, so can't np.array need to check whether that is an issue
        # or need to adapt structure
        setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
        if 'max' in setting_version_accumilator:
            if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['max'] = list()
            setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(list(all_settings_in_aug.values())), axis=0))
        if 'avg' in setting_version_accumilator:
            if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
            setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(list(all_settings_in_aug.values())), axis=0))
        aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)
        
             
    # For All Augs
    # For each sample, aggrigate across the augmentation
    final_combinations = dict()             # Shape: {combination: [sample, kld]}
    if 'max' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.max(np.array(stacked_kl_divs),axis=0).tolist()
    
    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            stacked_kl_divs = list()
            for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                stacked_kl_divs.append(aug_values[setting_accumilator])
                
            final_combinations[key] = np.mean(np.array(stacked_kl_divs),axis=0).tolist()
    
    if len(final_combinations) != 0:
        result['aggregated'] = dict()
        
    # Min-k
    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{metric_cfg.suffix}_kld_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                # Traditional divergence-MIA convention: higher top-k KLD/Renyi divergence -> higher score -> predicted member (no flip).
                sample_scores.append(float(np.mean(np.sort(sample)[-k_length:]))) 
            result['aggregated'][key] = sample_scores 
    return result, meta







# def renyi_kl_div_ripple_maxk(renyi_probs, metric_cfg, cfg, eps=1e-12):
#     result = dict()
#     meta = dict()
    
#     ratio = metric_cfg.ratio
#     _, aug_desc_dict = get_augmentations(cfg)
    
#     original_probs = renyi_probs['orig'][0]
#     number_of_samples = len(original_probs)
    
#     setting_version_accumilator = metric_cfg.augmentation_accumilator
#     aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

#     # Type: {max: [], avg: []}[augs]
#     aug_aggregated_per_sample_tokenwise_kl = list()
#     for aug, settings in renyi_probs.items():
#         if aug == "orig":
#             continue
        
#         all_settings_in_aug = dict()                                   #Shape: {setting: [sample, kld]}

#         # Processed Data
#         for setting_idx, setting in enumerate(settings):
#             key = aug_desc_dict[aug][setting_idx]
#             all_samples_in_setting_values = list()          # Shape: [samples, kld]
#             for sample_idx, aug_probs in enumerate(setting):
#                 org = torch.stack(original_probs[sample_idx]).float().cpu().numpy()
#                 org_log = np.log(org + eps)
#                 aug_log = np.log(torch.stack(aug_probs).float().cpu().numpy() + eps)

#                 kl = kl_div_per_token(org, org_log, aug_log) # KL: 1D vector
                
#                 tail_frac = getattr(metric_cfg, "tail_frac", 1/3)
#                 L = len(kl)
#                 tail_len = max(1, int(tail_frac * L)) if L > 0 else 0

#                 kl_tail = kl[-tail_len:] if tail_len > 0 else np.array([], dtype=float)

#                 all_samples_in_setting_values.append(kl_tail)
                
#             all_settings_in_aug[str(key)] = all_samples_in_setting_values
#             # all_settings_in_aug {setting: [sample, kld(1D)]}

#         # For Aug, Push sampled raw kl_array to meta return
#         meta[aug] = all_settings_in_aug
        
        
#         if aug not in result:
#             result[aug] = dict()
#         # Get Scores for Rawest kld values setting by setting no aggrigation
#         if 'none' in setting_version_accumilator:
#             for _ratio in ratio:
#                 for setting_name, setting_values in all_settings_in_aug.items():
#                     # key = f"Max_{_ratio}_{cfg.suffix}_no_agg_aug_{aug}_setting_{setting_idx}"
#                     key = ast.literal_eval(setting_name)
#                     key['k_ratio'] = _ratio
                                        
#                     sample_scores = list()
#                     # Cant use np.array because for some reason some samples have different sequence lengths, but even enventually for when using different description lengths
#                     # just keep it generalisabole to python lists
#                     for sample in setting_values:
#                         k_length = max(1, int(_ratio * len(sample)))
#                         # From the results of the LOGAN paper REGION II, we found that members have lower kld at the low std region, therefore we flip the score
#                         sample_scores.append((-1 * np.mean(np.sort(sample)[-k_length:])).item()) 
                        
#                     result[aug][str(key)] = sample_scores
                    
        
        
#         # AGGIGATION
#         # For each sample, aggrigate across the settings
#         #TODO: for some reason, all_settings_in_aug is inhomogeneous, so can't np.array need to check whether that is an issue
#         # or need to adapt structure
#         setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
#         if 'max' in setting_version_accumilator:
#             if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['max'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(all_settings_in_aug), axis=0))
#         if 'avg' in setting_version_accumilator:
#             if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(all_settings_in_aug), axis=0))
#         aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)
        
             
#     # For All Augs
#     # For each sample, aggrigate across the augmentation
#     final_combinations = dict()             # Shape: {combination: [sample, kld]}
#     if 'max' in aug_version_accumilator:
#         for setting_accumilator in setting_version_accumilator:
#             # Cannot aggrigate across augs if not aggrigated across setting per aug
#             if setting_accumilator == 'none': 
#                 continue
            
#             key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
#             stacked_kl_divs = list()
#             for aug_values in aug_aggregated_per_sample_tokenwise_kl:
#                 stacked_kl_divs.append(aug_values[setting_accumilator])
                
#             final_combinations[key] = np.max(np.array(stacked_kl_divs),axis=0).tolist()
    
#     if 'avg' in aug_version_accumilator:
#         for setting_accumilator in setting_version_accumilator:
#             # Cannot aggrigate across augs if not aggrigated across setting per aug
#             if setting_accumilator == 'none': 
#                 continue
            
#             key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
#             stacked_kl_divs = list()
#             for aug_values in aug_aggregated_per_sample_tokenwise_kl:
#                 stacked_kl_divs.append(aug_values[setting_accumilator])
                
#             final_combinations[key] = np.mean(np.array(stacked_kl_divs),axis=0).tolist()
    
#     if len(final_combinations) != 0:
#         result['aggregated'] = dict()
        
#     # Min-k
#     for _ratio in ratio:
#         for combination, samples in final_combinations.items():
#             key = f"Max_{_ratio}_{metric_cfg.suffix}_kld_{combination}"
#             sample_scores = list()
#             for sample in samples:
#                 k_length = max(1, int(_ratio * len(sample)))
#                 # From the results of the LOGAN paper REGION II, we found that members have lower kld at the low std region, therefore we flip the score
#                 sample_scores.append(float(-1 * np.mean(np.sort(sample)[-k_length:]))) 
#             result['aggregated'][key] = sample_scores 
#     return result, meta






# def renyi_divergence_ripple_maxk(probs, metric_cfg, cfg, eps=1e-12):
#     # print("Renyi-Div Metric")
#     alpha = metric_cfg.alpha
#     result = dict()
#     meta = dict()
    
#     ratio = metric_cfg.ratio
#     _, aug_desc_dict = get_augmentations(cfg)
    
#     original_probs = probs['orig'][0]
#     number_of_samples = len(original_probs)
    
#     setting_version_accumilator = metric_cfg.augmentation_accumilator
#     aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

#     # Type: {max: [], avg: []}[augs]
#     aug_aggregated_per_sample_tokenwise_kl = list()
#     for aug, settings in probs.items():
#         if aug == "orig":
#             continue
        
#         all_settings_in_aug = dict()                                     #Shape: [setting, sample, kld]

#          # Processed Data
#         for setting_idx, setting in enumerate(settings):
#             key = aug_desc_dict[aug][setting_idx]
#             all_samples_in_setting_values = list()          # Shape: [samples, kld]
#             for sample_idx, aug_probs in enumerate(setting):
#                 org = original_probs[sample_idx]
                
#                 kl = renyi_div_per_token(org, aug_probs, alpha, eps)
                
#                 tail_frac = getattr(metric_cfg, "tail_frac", 1/3)
#                 L = len(kl)
#                 tail_len = max(1, int(tail_frac * L)) if L > 0 else 0

#                 kl_tail = kl[-tail_len:] if tail_len > 0 else np.array([], dtype=float)

#                 all_samples_in_setting_values.append(kl_tail)
                
#             all_settings_in_aug[str(key)] = all_samples_in_setting_values
#             # all_settings_in_aug {setting: [sample, kld(1D)]}

#         # For Aug, Push sampled raw kl_array to meta return
#         meta[aug] = all_settings_in_aug
        
        
#         if aug not in result:
#             result[aug] = dict()
#         # Get Scores for Rawest kld values setting by setting no aggrigation
#         if 'none' in setting_version_accumilator:
#             for _ratio in ratio:
#                 for setting_name, setting_values in all_settings_in_aug.items():
#                     # key = f"Max_{_ratio}_{cfg.suffix}_no_agg_aug_{aug}_setting_{setting_idx}"
#                     key = ast.literal_eval(setting_name)
#                     key['k_ratio'] = _ratio
                                        
#                     sample_scores = list()
#                     # Cant use np.array because for some reason some samples have different sequence lengths, but even enventually for when using different description lengths
#                     # just keep it generalisabole to python lists
#                     for sample in setting_values:
#                         k_length = max(1, int(_ratio * len(sample)))
#                         # From the results of the LOGAN paper REGION II, we found that members have lower kld at the low std region, therefore we flip the score
#                         sample_scores.append((-1 * np.mean(np.sort(sample)[-k_length:])).item()) 
                        
#                     result[aug][str(key)] = sample_scores
        
#         # AGGIGATION
#         # For each sample, aggrigate across the settings
#         #TODO: for some reason, all_settings_in_aug is inhomogeneous, so can't np.array need to check whether that is an issue
#         # or need to adapt structure
#         setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
#         if 'max' in setting_version_accumilator:
#             if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['max'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(all_settings_in_aug), axis=0))
#         if 'avg' in setting_version_accumilator:
#             if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(all_settings_in_aug), axis=0))
#         aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)
        
             
#     # For All Augs
#     # For each sample, aggrigate across the augmentation
#     final_combinations = dict()             # Shape: {combination: [sample, kld]}
#     if 'max' in aug_version_accumilator:
#         for setting_accumilator in setting_version_accumilator:
#             # Cannot aggrigate across augs if not aggrigated across setting per aug
#             if setting_accumilator == 'none': 
#                 continue
            
#             key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
#             stacked_kl_divs = list()
#             for aug_values in aug_aggregated_per_sample_tokenwise_kl:
#                 stacked_kl_divs.append(aug_values[setting_accumilator])
                
#             final_combinations[key] = np.max(np.array(stacked_kl_divs),axis=0).tolist()
    
#     if 'avg' in aug_version_accumilator:
#         for setting_accumilator in setting_version_accumilator:
#             # Cannot aggrigate across augs if not aggrigated across setting per aug
#             if setting_accumilator == 'none': 
#                 continue
            
#             key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
#             stacked_kl_divs = list()
#             for aug_values in aug_aggregated_per_sample_tokenwise_kl:
#                 stacked_kl_divs.append(aug_values[setting_accumilator])
                
#             final_combinations[key] = np.mean(np.array(stacked_kl_divs),axis=0).tolist()
    
#     if len(final_combinations) != 0:
#         result['aggregated'] = dict()
        
#     # Min-k
#     for _ratio in ratio:
#         for combination, samples in final_combinations.items():
#             key = f"Max_{_ratio}_{metric_cfg.suffix}_kld_{combination}"
#             sample_scores = list()
#             for sample in samples:
#                 k_length = max(1, int(_ratio * len(sample)))
#                 # From the results of the LOGAN paper REGION II, we found that members have lower kld at the low std region, therefore we flip the score
#                 sample_scores.append(float(-1 * np.mean(np.sort(sample)[-k_length:]))) 
#             result['aggregated'][key] = sample_scores 
#     return result, meta


def renyi_div_per_token(org_probs, aug_probs, alpha, eps=1e-12):
    org_probs = np.clip(org_probs, eps, 1.0)
    aug_probs = np.clip(aug_probs, eps, 1.0)
    divergence = (1 / (alpha - 1)) * np.log(np.sum(org_probs**alpha * aug_probs**(1 - alpha), axis=1) + eps)
    return divergence





def get_token_slice(seq, slice_idx, num_slices):
    n = len(seq)
    if num_slices <= 1:
        return seq
    if n == 0:
        return []

    # Clamp slice_idx just in case
    slice_idx = max(0, min(slice_idx, num_slices - 1))

    # Even split: base size + distribute remainder to early slices
    base = n // num_slices
    rem = n % num_slices

    # First `rem` slices get (base + 1), rest get base
    start = slice_idx * base + min(slice_idx, rem)
    end = start + base + (1 if slice_idx < rem else 0)

    return seq[start:end]