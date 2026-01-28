import numpy as np
from collections import defaultdict
import torch

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
            result[result_key].append(-1*np.mean(_loss_diff[_key]).item())

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
            result[result_key].append(np.mean(_loss_diff[_key]).item()) # members has smaller score (later will be flipped.)

    return result

def renyi_kl_div_maxk(renyi_probs, cfg, eps=1e-12):
    # print("KL-Div Metric")
    result = dict()
    meta = dict()
    
    ratio = cfg.ratio
    original_probs = renyi_probs['orig'][0]
    number_of_samples = len(original_probs)
    # print(f'len(original_probs): {number_of_samples}')
    
    setting_version_accumilator = cfg.augmentation_accumilator
    aug_version_accumilator = cfg.augmentation_setting_version_accumilator

    # Type: {max: [], avg: []}[augs]
    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in renyi_probs.items():
        if aug == "orig":
            continue
        
        all_raw_metric_values = [[] for _ in range(number_of_samples)] #Shape: [samples, setting, kld]
        all_settings_in_aug = list()                                   #Shape: [setting, sample, kld]

        # Processed Data
        for setting_idx, setting in enumerate(settings):
            # print(f'Length of settings: {len(setting)}')
            all_samples_in_setting_values = list()          # Shape: [samples, kld]
            for sample_idx, aug_probs in enumerate(setting):
                # print(f"sample_idx: {sample_idx}")
                org = torch.stack(original_probs[sample_idx]).float().cpu().numpy()
                org_log = np.log(org + eps)
                aug_log = np.log(torch.stack(aug_probs).float().cpu().numpy() + eps)

                kl = kl_div_per_token(org, org_log, aug_log) # KL: 1D vector
                
                # Append Values to Respective Data Storage
                all_raw_metric_values[sample_idx].append(kl.tolist())
                all_samples_in_setting_values.append(kl)
                
            all_settings_in_aug.append(all_samples_in_setting_values)
            # all_settings_in_aug [setting, sample, kld(1D)]

        # For Aug, Push sampled raw kl_array to meta return
        meta[aug] = all_raw_metric_values
        
        # Get Scores for Rawest kld values setting by setting no aggrigation
        if 'none' in setting_version_accumilator or 'none' in aug_version_accumilator:
            for _ratio in ratio:
                for setting_idx, setting_values in enumerate(all_settings_in_aug):
                    key = f"Max_{_ratio}_{cfg.suffix}_no_agg_aug_{aug}_setting_{setting_idx}"
                    sample_scores = list()
                    # Cant use np.array because for some reason some samples have different sequence lengths, but even enventually for when using different description lengths
                    # just keep it generalisabole to python lists
                    for sample in setting_values:
                        k_length = max(1, int(_ratio * len(sample)))
                        sample_scores.append((-1 * np.mean(np.sort(sample)[-k_length:])).item())
                        
                    result[key] = sample_scores
                    
        
        
        # AGGIGATION
        # For each sample, aggrigate across the settings
        setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
        if 'max' in setting_version_accumilator:
            if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['max'] = list()
            for sample in all_raw_metric_values:
                setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(sample), axis=0))
        if 'avg' in setting_version_accumilator:
            if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
            for sample in all_raw_metric_values:
                setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(sample), axis=0))
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
            if key not in final_combinations:
                final_combinations[key] = list()
            for sample_idx in range(number_of_samples):
                # [aug, kld]
                sample_kld = list()
                for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                    sample_kld.append(aug_values[setting_accumilator][sample_idx])

                final_combinations[key].append((np.max(np.array(sample_kld), axis=0)).tolist())
    
    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            if key not in final_combinations:
                final_combinations[key] = list()
            for sample_idx in range(number_of_samples):
                # [aug, kld]
                sample_kld = list()
                for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                    sample_kld.append(aug_values[setting_accumilator][sample_idx])

                final_combinations[key].append(np.mean(np.array(sample_kld), axis=0).tolist())
                    
        
    # Min-k
    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{cfg.suffix}_kld_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                sample_scores.append(float(-1 * np.mean(np.sort(sample)[-k_length:])))
            result[key] = sample_scores
            
            
    return result, meta


def kl_div_per_token(org_probs, org_log_probs, aug_log_probs):
    return np.sum(org_probs * (org_log_probs - aug_log_probs),axis=1)
    





def renyi_divergence_maxk(probs, cfg):
    # print("Renyi-Div Metric")
    alpha = cfg.alpha
    result = dict()
    meta = dict()
    
    ratio = cfg.ratio
    original_probs = probs['orig'][0]
    number_of_samples = len(original_probs)
    
    setting_version_accumilator = cfg.augmentation_accumilator
    aug_version_accumilator = cfg.augmentation_setting_version_accumilator

    # Type: {max: [], avg: []}[augs]
    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in probs.items():
        if aug == "orig":
            continue
        
        all_raw_metric_values = [[] for _ in range(number_of_samples)] #Shape: [samples, setting, kld]
        all_settings_in_aug = list()                                   #Shape: [setting, sample, kld]

        # Processed Data
        for setting_idx, setting in enumerate(settings):
            all_samples_in_setting_values = list()          # Shape: [samples, kld]
            for sample_idx, aug_probs in enumerate(setting):
                kl = renyi_div_per_token(np.stack(original_probs[sample_idx]),
                                         np.stack(aug_probs),
                                         alpha,
                                         1e-12)
                
                # Append Values to Respective Data Storage
                all_raw_metric_values[sample_idx].append(kl.tolist())
                all_samples_in_setting_values.append(kl)
                
            all_settings_in_aug.append(all_samples_in_setting_values)
        
        # For Aug, Push sampled raw kl_array to meta return
        meta[aug] = all_raw_metric_values
        
        # Get Scores for Rawest kld values setting by setting no aggrigation
        if 'none' in setting_version_accumilator or 'none' in aug_version_accumilator:
            for _ratio in ratio:
                for setting_idx, setting_values in enumerate(all_settings_in_aug):
                    key = f"Max_{_ratio}_{cfg.suffix}_no_agg_aug_{aug}_setting_{setting_idx}"
                    sample_scores = list()
                    # Cant use np.array because for some reason some samples have different sequence lengths, but even enventually for when using different description lengths
                    # just keep it generalisabole to python lists
                    for sample in setting_values:
                        k_length = max(1, int(_ratio * len(sample)))
                        sample_scores.append((-1 * np.mean(np.sort(sample)[-k_length:])).item())
                        
                    result[key] = sample_scores
                    
        
        
        # AGGIGATION
        # For each sample, aggrigate across the settings
        setting_aggregated_per_sample_tokenwise_kl = dict() # Shape: {max: [sample, kld], avg: [sample, kld]}
        if 'max' in setting_version_accumilator:
            if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['max'] = list()
            for sample in all_raw_metric_values:
                setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(np.array(sample), axis=0))
        if 'avg' in setting_version_accumilator:
            if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
                setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
            for sample in all_raw_metric_values:
                setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(np.array(sample), axis=0))
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
            if key not in final_combinations:
                final_combinations[key] = list()
            for sample_idx in range(number_of_samples):
                # [aug, kld]
                sample_kld = list()
                for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                    sample_kld.append(aug_values[setting_accumilator][sample_idx])

                final_combinations[key].append((np.max(np.array(sample_kld), axis=0)).tolist())
    
    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            # Cannot aggrigate across augs if not aggrigated across setting per aug
            if setting_accumilator == 'none': 
                continue
            
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            if key not in final_combinations:
                final_combinations[key] = list()
            for sample_idx in range(number_of_samples):
                # [aug, kld]
                sample_kld = list()
                for aug_values in aug_aggregated_per_sample_tokenwise_kl:
                    sample_kld.append(aug_values[setting_accumilator][sample_idx])

                final_combinations[key].append(np.mean(np.array(sample_kld), axis=0).tolist())
                    
        
    # Min-k
    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{cfg.suffix}_renyi_div_alpha_{alpha}_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                sample_scores.append(float(-1 * np.mean(np.sort(sample)[-k_length:])))
            result[key] = sample_scores
            
            
    return result, meta

















    
# def renyi_divergence_mink(probs,sampled_indices, cfg):
#     print("Renyi-Div Metric")
#     ratio = cfg.ratio
#     alpha = cfg.alpha
#     result = dict()
#     meta = dict()
#     original_probs = probs['orig'][0]

#     setting_version_accumilator = cfg.augmentation_accumilator
#     aug_version_accumilator = cfg.augmentation_setting_version_accumilator

#     aggregated_values = dict()
#     setting_aggregated_per_sample_tokenwise_kl = dict()

#     for aug, settings in probs.items():
#         if aug == "orig":
#             continue

#         # Processed Data
#         all_setting_in_aug_values = list()      #divergence values for all settings and all samples in the augmentation: (settings, sample, kld)
#         sampled_setting_in_aug_values = list()     #divergence values for all settings and sampled samples in the augmentation: (settings, sample, kld)
#         for setting_idx, setting in enumerate(settings):
            
#             # Data Storage for Current Setting
#             all_sample_in_setting_values = list()
#             sampled_sample_in_setting_values = list()
            
#             for sample_idx, aug_probs in enumerate(setting):
#                 kl = renyi_div_per_token(np.stack(original_probs[sample_idx]),
#                                          np.stack(aug_probs),
#                                          alpha,
#                                          1e-12)
                
#                 # There is an issue with some samples having length of 576 elements just cut the end off
#                 if(len(kl) != 576):
#                     kl = kl[:576]
                
#                 # Append Values to Respective Data Storage
#                 all_sample_in_setting_values.append(kl)
#                 if sample_idx in sampled_indices:
#                     sampled_sample_in_setting_values.append(kl)
                    
#             # Append the setting's values to parent aug
#             all_setting_in_aug_values.append(all_sample_in_setting_values)
#             sampled_setting_in_aug_values.append(sampled_sample_in_setting_values)
        
#         # For Aug, Create nparray and send meta values to aug key
#         final_values = np.array(all_setting_in_aug_values)
#         meta[aug] = np.array(sampled_setting_in_aug_values)
        
#         # Within Each Aug Aggrigate the Settings Accordingly
#         if 'max' in setting_version_accumilator:
#             if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['max'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(final_values, axis=0))
            
#         if 'avg' in setting_version_accumilator:
#             if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(final_values, axis=0))
             
#     # For Each way settings are aggreigated, aggrigate augmentations accordingly
#     for setting_aggrigation, values in setting_aggregated_per_sample_tokenwise_kl.items():
#         if 'max' in aug_version_accumilator:
#             aggregated_values[f"maxed_aug_{setting_aggrigation}ed_settings"] = np.max(np.array(values), axis=0)
#             meta[f'final_agg_maxed_aug_{setting_aggrigation}ed_settings'] = np.max(np.array(values), axis=0)
#         if 'avg' in aug_version_accumilator:
#             aggregated_values[f"avged_aug_{setting_aggrigation}ed_settings"] = np.mean(np.array(values), axis=0)
#             meta[f'final_agg_avged_aug_{setting_aggrigation}ed_settings'] = np.max(np.array(values), axis=0)
        
#     # Min-k
#     for _ratio in ratio:
#         for combination, value in aggregated_values.items():
#             key = f"Min_{_ratio}_{cfg.suffix}_renyi_div_alpha_{alpha}_{combination}"
#             k_length = max(1, int(_ratio * len(value)))
#             result[key] = (-1 * np.mean(np.sort(value, axis=1)[:, -k_length:], axis=1)).tolist()
            
#     return result, meta    
    

    
    
    
    
# def renyi_divergence_mink(probs,sampled_indices, cfg):
#     print("Renyi-Div Metric")
#     ratio = cfg.ratio
#     alpha = cfg.alpha
#     result = dict()
#     meta = dict()
#     original_probs = probs['orig'][0]

    
#     setting_version_accumilator = cfg.augmentation_accumilator
#     aug_version_accumilator = cfg.augmentation_setting_version_accumilator

#     per_sample_kld = defaultdict(list)
#     setting_aggregated_per_sample_tokenwise_kl = defaultdict(lambda: defaultdict(list))

#     for aug, settings in probs.items():
#         if aug == "orig":
#             continue
        
#         # Meta Metrics for All Settings (and samples per setting) in this augmentation (in this batch)
#         all_setting_in_aug_values = list()
#         per_sample_tokenwise_kl = defaultdict(list)
#         for setting_idx, setting in enumerate(settings):
#             # Meta Metrics for all samples in this estting
#             all_sample_in_setting_values = list()
#             for sample_idx, aug_probs in enumerate(setting):
#                 kl = renyi_div_per_token(np.stack(original_probs[sample_idx]),
#                                          np.stack(aug_probs),
#                                          alpha,
#                                          1e-12)
                
#                 print(f"For Sample {sample_idx}, kl length: {len(kl)}")
                
#                 per_sample_tokenwise_kl[sample_idx].append(kl)
                
#                 # For raw meta metics
#                 all_sample_in_setting_values.append(kl.tolist())
#             all_setting_in_aug_values.append(all_sample_in_setting_values)
            
            
#         meta[aug] = np.array(all_setting_in_aug_values)


#         for idx, sample in per_sample_tokenwise_kl.items():
#             if 'max' in setting_version_accumilator:
#                 setting_aggregated_per_sample_tokenwise_kl['max'][idx].append(
#                     np.max(np.array(sample), axis=0)
#                 )
#             if 'avg' in setting_version_accumilator:
#                 setting_aggregated_per_sample_tokenwise_kl['avg'][idx].append(
#                     np.mean(np.array(sample), axis=0)
#                 )

#     for setting_accumilator, samples in setting_aggregated_per_sample_tokenwise_kl.items():
#         for sample_idx, sample_kld in samples.items():
#             if 'max' in aug_version_accumilator:
#                 per_sample_kld[f"maxed_aug_{setting_accumilator}ed_settings"].append(
#                     np.max(np.array(sample_kld), axis=0)
#                 )
#             if 'avg' in aug_version_accumilator:
#                 per_sample_kld[f"avged_aug_{setting_accumilator}ed_settings"].append(
#                     np.mean(np.array(sample_kld), axis=0)
#                 )

#     # Min-K across samples for each combination
#     for _ratio in ratio:
#         for combination, samples in per_sample_kld.items():
#             key = f"Min_{_ratio}_renyi_div_alpha_{alpha}_{combination}"
#             min_k_kl = list()
#             for sample in samples:
#                 k_length = max(1, int(_ratio * len(sample)))
#                 min_k_kl.append(float(-1 * np.mean(np.sort(sample)[-k_length:])))
#             result[key] = min_k_kl

#     return result, meta



def renyi_div_per_token(org_probs, aug_probs, alpha, eps=1e-12):
    org_probs = np.clip(org_probs, eps, 1.0)
    aug_probs = np.clip(aug_probs, eps, 1.0)
    divergence = (1 / (alpha - 1)) * np.log(np.sum(org_probs**alpha * aug_probs**(1 - alpha), axis=1) + eps)
    return divergence













# PREVIOUS CODE:

'''
Originally used nparray and stack latter on, but found issues with some token sequences being longer than others which is strange
so initally cut it off but this isn't generalisable for future unlimited description lengths so revert to not using np.array
'''

# def renyi_kl_div_mink(renyi_probs, sampled_indices, cfg, eps=1e-12):
#     print("KL-Div Metric")
#     ratio = cfg.ratio
#     result = dict()
#     meta = dict()
#     original_probs = renyi_probs['orig'][0]

#     setting_version_accumilator = cfg.augmentation_accumilator
#     aug_version_accumilator = cfg.augmentation_setting_version_accumilator

#     aggregated_values = dict()
#     setting_aggregated_per_sample_tokenwise_kl = dict()

#     for aug, settings in renyi_probs.items():
#         if aug == "orig":
#             continue

#         # Processed Data
#         all_setting_in_aug_values = list()      #kld values for all settings and all samples in the augmentation: (settings, sample, kld)
#         sampled_setting_in_aug_values = list()     #kld values for all settings and sampled samples in the augmentation: (settings, sample, kld)
#         for setting_idx, setting in enumerate(settings):
            
#             # Data Storage for Current Setting
#             all_sample_in_setting_values = list()
#             sampled_sample_in_setting_values = list()
            
#             for sample_idx, aug_probs in enumerate(setting):
#                 org = torch.stack(original_probs[sample_idx]).cpu().numpy()
#                 org_log = np.log(org + eps)
#                 aug_log = np.log(torch.stack(aug_probs).cpu().numpy() + eps)

#                 kl = kl_div_per_token(org, org_log, aug_log)
                
#                 # There is an issue with some samples having length of 576 elements just cut the end off
#                 if(len(kl) != 575):
#                     kl = kl[:575]
                
#                 # Append Values to Respective Data Storage
#                 all_sample_in_setting_values.append(kl)
#                 if sample_idx in sampled_indices:
#                     sampled_sample_in_setting_values.append(kl)
                    
#             # Append the setting's values to parent aug
#             all_setting_in_aug_values.append(all_sample_in_setting_values)
#             sampled_setting_in_aug_values.append(sampled_sample_in_setting_values)
        
#         # For Aug, Create nparray and send meta values to aug key
#         final_values = np.array(all_setting_in_aug_values)
#         meta[aug] = np.array(sampled_setting_in_aug_values)
        
#         # Within Each Aug Aggrigate the Settings Accordingly
#         if 'max' in setting_version_accumilator:
#             if 'max' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['max'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(final_values, axis=0))
            
#         if 'avg' in setting_version_accumilator:
#             if 'avg' not in setting_aggregated_per_sample_tokenwise_kl:
#                 setting_aggregated_per_sample_tokenwise_kl['avg'] = list()
#             setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(final_values, axis=0))
             
#     # For Each way settings are aggreigated, aggrigate augmentations accordingly
#     for setting_aggrigation, values in setting_aggregated_per_sample_tokenwise_kl.items():
#         if 'max' in aug_version_accumilator:
#             aggregated_values[f"final_agg_maxed_aug_{setting_aggrigation}ed_settings"] = np.max(np.array(values), axis=0)
#         if 'avg' in aug_version_accumilator:
#             aggregated_values[f"final_agg_avged_aug_{setting_aggrigation}ed_settings"] = np.mean(np.array(values), axis=0)
        
        
#     # Min-k
#     for _ratio in ratio:
#         for combination, value in aggregated_values.items():
#             key = f"Min_{_ratio}_{cfg.suffix}_kld_{combination}"
#             k_length = max(1, int(_ratio * len(value)))
#             result[key] = (-1 * np.mean(np.sort(value, axis=1)[:, -k_length:], axis=1)).tolist()
            
#     return result, meta
