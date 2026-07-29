
import torch
import numpy as np
import sys

def renyi_probs(token_probs_clamped, alpha):
    if alpha == "inf":
        # For α = ∞, we want to create a one-hot vector with the maximum probability
        max_values, max_indices = torch.max(token_probs_clamped, dim=-1, keepdim=True)
        # Create a zero tensor with the same shape
        renyi_normalized = torch.zeros_like(token_probs_clamped)
        # Set the maximum value to 1 (one-hot encoding)
        renyi_normalized.scatter_(-1, max_indices, 1.0)
        return renyi_normalized
    else:
        renyi_numerator = torch.pow(token_probs_clamped, alpha)
        renyi_denominator = torch.sum(renyi_numerator, dim=-1, keepdim=True)
        renyi_normalized = renyi_numerator / renyi_denominator
        return renyi_normalized


def get_meta_metrics_by_part(total_parts, part, cfg):
    """
    input_ids : mix_input_ids
    attention_masks : mix_attention_masks

    total_parts :
    {
        "orig" : 
        {   
            [{
                "img" : {
                    "logits : [
                        (sample 1 img logits),
                        (sample 2 img logits)
                    ],
                    "probabilities" : [
                        (sample 1 img probs),
                        (sample 2 img probs)
                    ]
                    "log_probabilities" : [
                        (sample 1 img log_probs),
                        (sample 2 img log_probs),
                    ]
                },
                "inst": {
                    "logits" : [],
                    "probabilities : [],
                    "log_probabilities : []
                }
            }]
        }    
        "aug_1" : [
            {...},
            {...}
        ]
    }

    If not augmented, parts dict only has a "orig" key.

    meta_metrics: {
        "ppl": {
            "aug_type" : [
                    [(sample 1 loss values), (sample 2 loss values)...]
                    [(sample 1 loss values), (sample 2 loss values)...]
                ]
            }
        }
    }
    """

    meta_metrics = dict()
    meta_metrics["ppl"] = dict()
    meta_metrics["entropies"] = dict()
    meta_metrics["all_prob"] = dict()
    meta_metrics["probabilities"] = dict()
    meta_metrics["log_probabilities"] = dict()
    meta_metrics["modified_entropies"] = dict()
    meta_metrics["max_probs"] = dict()
    meta_metrics["gap_probs"] = dict()
    meta_metrics["renyi_05_entro"] = dict()
    meta_metrics["renyi_2_entro"] = dict()
    meta_metrics["losses"] = dict()
    meta_metrics["modified_entropies"] = dict()
    meta_metrics["modified_entropies_alpha_05"] = dict()
    meta_metrics["modified_entropies_alpha_2"] = dict()
    meta_metrics["no_norm_probs"] = dict()
    meta_metrics["renyi_05_probs"] = dict()
    meta_metrics["renyi_1_probs"] = dict()
    meta_metrics["renyi_2_probs"] = dict()
    meta_metrics["renyi_inf_probs"] = dict()
    
    # Here add our metrics
    meta_metrics["per_token_CE_loss"] = dict()

    epsilon = 1e-4

    for aug_type, aug_results in total_parts.items():
        meta_metrics["ppl"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["entropies"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["all_prob"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["probabilities"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["log_probabilities"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["modified_entropies"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["max_probs"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["gap_probs"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["renyi_05_entro"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["renyi_2_entro"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["losses"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["modified_entropies"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["modified_entropies_alpha_05"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["modified_entropies_alpha_2"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["no_norm_probs"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["renyi_05_probs"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["renyi_1_probs"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["renyi_2_probs"][aug_type] = [[] for _ in range(len(aug_results))]
        meta_metrics["renyi_inf_probs"][aug_type] = [[] for _ in range(len(aug_results))]

        # Add our metrics
        meta_metrics["per_token_CE_loss"][aug_type] = [[] for _ in range(len(aug_results))]

        for aug_idx, aug_result in enumerate(aug_results):
            # If a metric is obtained using the entire logits/input_ids, use list()
            # If a metric is obtained for each token_position, use [[] for _ in range(len(aug_result[part]["input_ids"]))]
            
            # print('aug_result:')
            # print(aug_result)
            

            
            meta_metrics["ppl"][aug_type][aug_idx] = list()
            meta_metrics["entropies"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["all_prob"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["probabilities"][aug_type][aug_idx] = list()
            meta_metrics["log_probabilities"][aug_type][aug_idx] = list()
            meta_metrics["modified_entropies"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["max_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["gap_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["renyi_05_entro"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["renyi_2_entro"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["losses"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["modified_entropies"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["modified_entropies_alpha_05"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["modified_entropies_alpha_2"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["no_norm_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]         # 2d array (token_seq, vocab_size)
            meta_metrics["renyi_05_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]         # 2d array (token_seq, vocab_size)
            meta_metrics["renyi_1_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["renyi_2_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]
            meta_metrics["renyi_inf_probs"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]

            # Add our metrics
            meta_metrics["per_token_CE_loss"][aug_type][aug_idx] = [[] for _ in range(len(aug_result[part]["input_ids"]))]

            for _batch_idx in range(len(aug_result[part]["input_ids"])):
                for _token_idx, token_id in enumerate(aug_result[part]["input_ids"][_batch_idx][1:]):
                    # Theis is where the per-token metrics are computed
                    # If a meta_metric is a tensor, set them to numpy array.
                    
                    token_probs = aug_result[part]["probabilities"][_batch_idx][_token_idx, :]
                    token_log_probs = aug_result[part]["log_probabilities"][_batch_idx][_token_idx, :]
                    
                    token_probs_clamped = torch.clamp(token_probs, min=epsilon, max=1-epsilon)
                    token_log_probs_clamped = token_probs_clamped.log()
                    
                    # Renyi_1
                    entropy = -(token_probs * token_log_probs_clamped).sum().item()
                    
                    meta_metrics["entropies"][aug_type][aug_idx][_batch_idx].append(entropy)
                    meta_metrics["renyi_1_probs"][aug_type][aug_idx][_batch_idx].append(renyi_probs(token_probs_clamped, 1).detach().float().cpu())
                    
                    if (
                        "max_k_no_norn_kl_div" in cfg.img_metrics.metrics_to_use
                        or "max_k_no_norn_kl_div_tkn_vals" in cfg.img_metrics.get_proc_meta_metrics
                        or "no_norm_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_k_no_norn_kl_div_ripple" in cfg.img_metrics.metrics_to_use
                        or "max_k_no_norn_kl_div_tkn_vals_ripple" in cfg.img_metrics.get_proc_meta_metrics
                        or "entropy_gated_no_norn_kl_div" in cfg.img_metrics.metrics_to_use
                    ):
                        # No_norm
                        meta_metrics["no_norm_probs"][aug_type][aug_idx][_batch_idx].append(token_probs_clamped.detach().float().cpu())

                    # Renyi_05
                    if (
                        "max_k_renyi_05_kl_div" in cfg.img_metrics.metrics_to_use
                        or "mod_renyi_05_entro" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_05_entro" in cfg.img_metrics.metrics_to_use
                        or "min_k_renyi_05_entro" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_05_kl_div_tkn_vals" in cfg.img_metrics.get_proc_meta_metrics
                        or "renyi_05_entro" in cfg.img_metrics.get_raw_meta_metrics
                        or "renyi_05_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_k_renyi_05_kl_div_ripple" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_05_kl_div_tkn_vals_ripple" in cfg.img_metrics.get_proc_meta_metrics
                        or "entropy_gated_renyi_05_kl_div" in cfg.img_metrics.metrics_to_use
                    ):
                        alpha=0.5
                        renyi_05 = (1 / (1-alpha)) * torch.log(torch.sum(torch.pow(token_probs_clamped, alpha))).item()
                        meta_metrics["renyi_05_entro"][aug_type][aug_idx][_batch_idx].append(renyi_05)
                        meta_metrics["renyi_05_probs"][aug_type][aug_idx][_batch_idx].append(renyi_probs(token_probs_clamped, 0.5).detach().float().cpu())

                    # Renyi_2
                    if (
                        "max_k_renyi_2_kl_div" in cfg.img_metrics.metrics_to_use
                        or "mod_renyi_2_entro" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_2_entro" in cfg.img_metrics.metrics_to_use
                        or "min_k_renyi_2_entro" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_2_kl_div_tkn_vals" in cfg.img_metrics.get_proc_meta_metrics
                        or "renyi_2_entro" in cfg.img_metrics.get_raw_meta_metrics
                        or "renyi_2_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_k_renyi_2_kl_div_ripple" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_2_kl_div_tkn_vals_ripple" in cfg.img_metrics.get_proc_meta_metrics
                        or "entropy_gated_renyi_2_kl_div" in cfg.img_metrics.metrics_to_use
                    ):
                        alpha=2
                        renyi_2 = (1 / (1-alpha)) * torch.log(torch.sum(torch.pow(token_probs_clamped, alpha))).item()
                        meta_metrics["renyi_2_entro"][aug_type][aug_idx][_batch_idx].append(renyi_2)
                        meta_metrics["renyi_2_probs"][aug_type][aug_idx][_batch_idx].append(renyi_probs(token_probs_clamped, 2).detach().float().cpu())

                    # Renyi_inf
                    if (
                        "max_k_renyi_inf_kl_div" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_inf_kl_div_tkn_vals" in cfg.img_metrics.get_proc_meta_metrics
                        or "gap_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "renyi_inf_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_k_renyi_inf_kl_div_ripple" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_inf_kl_div_tkn_vals_ripple" in cfg.img_metrics.get_proc_meta_metrics
                        or "entropy_gated_renyi_inf_kl_div" in cfg.img_metrics.metrics_to_use
                    ):
                        max_p = token_log_probs_clamped.max().item()
                        second_p = token_log_probs_clamped[token_log_probs_clamped != token_log_probs_clamped.max()].max().item()
                        gap_p = max_p - second_p
                        meta_metrics["gap_probs"][aug_type][aug_idx][_batch_idx].append(gap_p)
                        meta_metrics["max_probs"][aug_type][aug_idx][_batch_idx].append(max_p)
                        meta_metrics["renyi_inf_probs"][aug_type][aug_idx][_batch_idx].append(renyi_probs(token_probs_clamped, "inf").detach().float().cpu())

                    if (
                        "mink" in cfg.img_metrics.metrics_to_use
                        or "all_prob" in cfg.img_metrics.get_raw_meta_metrics
                        or "losses" in cfg.img_metrics.get_raw_meta_metrics
                    ):
                        min_k_p = token_log_probs_clamped[token_id].item()
                        meta_metrics["all_prob"][aug_type][aug_idx][_batch_idx].append(min_k_p)
                        cross_entropy_loss = -min_k_p
                        meta_metrics["losses"][aug_type][aug_idx][_batch_idx].append(cross_entropy_loss)

                    # Modified entropy
                    if(
                        "mod_renyi_1_entro" in cfg.img_metrics.metrics_to_use
                        or "mod_renyi_05_entro" in cfg.img_metrics.metrics_to_use
                        or "mod_renyi_2_entro" in cfg.img_metrics.metrics_to_use
                        or "modified_entropies" in cfg.img_metrics.get_raw_meta_metrics
                    ):
                        p_y = token_probs_clamped[token_id].item()
                        modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs_clamped * torch.log(1 - token_probs_clamped)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
                        meta_metrics["modified_entropies"][aug_type][aug_idx][_batch_idx].append(modified_entropy)

                    token_probs_remaining = torch.cat((token_probs_clamped[:token_id], token_probs_clamped[token_id+1:]))
                    
                    if(
                        "mod_renyi_05_entro" in cfg.img_metrics.metrics_to_use
                        or "mod_renyi_2_entro" in cfg.img_metrics.metrics_to_use
                        or "modified_entropies_alpha_05" in cfg.img_metrics.get_raw_meta_metrics
                        or "modified_entropies_alpha_2" in cfg.img_metrics.get_raw_meta_metrics
                    ):
                        for alpha in [0.5,2]:
                            entropy = - (1 / abs(1 - alpha)) * (
                                (1-p_y)* p_y**(abs(1-alpha))\
                                    - (1-p_y)
                                    + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                                    - torch.sum(token_probs_remaining)
                                    ).item() 
                            if alpha==0.5:
                                meta_metrics["modified_entropies_alpha_05"][aug_type][aug_idx][_batch_idx].append(entropy)
                            if alpha==2:
                                meta_metrics["modified_entropies_alpha_2"][aug_type][aug_idx][_batch_idx].append(entropy)

                    # Add our metrics
                    if(
                        "cross_entropy_mink" in cfg.img_metrics.metrics_to_use
                        or "cross_entropy_diff_mink" in cfg.img_metrics.metrics_to_use
                        or "per_token_CE_loss" in cfg.img_metrics.get_raw_meta_metrics
                    ):
                        _ce_loss = -token_log_probs[token_id].float().cpu().numpy()
                        meta_metrics["per_token_CE_loss"][aug_type][aug_idx][_batch_idx].append(_ce_loss)

                if(
                    "losses" in cfg.img_metrics.get_raw_meta_metrics
                ):
                    loss = np.nanmean(meta_metrics["losses"][aug_type][aug_idx][_batch_idx])
                    
                if(
                    "ppl" in cfg.img_metrics.get_raw_meta_metrics
                ):
                    meta_metrics["ppl"][aug_type][aug_idx].append(np.exp(loss))
                
                if(
                    "aug_kl" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_025" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_05" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_2" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_4" in cfg.img_metrics.metrics_to_use
                    or "probabilities" in cfg.img_metrics.get_raw_meta_metrics
                    or "max_k_renyi_divergence_025_ripple" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_05_ripple" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_2_ripple" in cfg.img_metrics.metrics_to_use
                    or "max_k_renyi_divergence_4_ripple" in cfg.img_metrics.metrics_to_use
                ):
                    meta_metrics["probabilities"][aug_type][aug_idx].append(aug_result[part]["probabilities"][_batch_idx].float().cpu().numpy())
                
                if(
                    "aug_kl" in cfg.img_metrics.metrics_to_use
                    or "log_probabilities" in cfg.img_metrics.get_raw_meta_metrics
                ):
                    meta_metrics["log_probabilities"][aug_type][aug_idx].append(aug_result[part]["log_probabilities"][_batch_idx].float().cpu().numpy())
    
    return meta_metrics
