import torch
import numpy as np
from src.metrics.baseline_metrics import aug_kl, min_k, mod_entropy, mod_renyi, max_prob_gap, max_entropy, min_entropy
from src.metrics.proposed_metrics import cross_entropy_mink, cross_entropy_diff_mink, renyi_kl_div_maxk, renyi_divergence_maxk

def get_img_metric_by_parts(meta_metrics, cfg):
    """
    meta_metrics: batched or full meta_metrics dictionary
    meta_metrics : {
        "ppl" : {
            "aug_1" : [
                [sample1, sample2, sample3],
                [sample1, sample2, sample3]
            ],
            "aug_2" : [
            
            ]
        },
        "entropies": {
            "aug_1" : [
                [    
                    [ entropy1, entropy2, ...]
                ]      
            ] 
        }
        
    }

    part is not necessary because it was already processed from the get_meta_metrics_by_part function.
    """

    pred = dict()
    meta = dict()

    if "aug_kl" in cfg.img_metrics.metrics_to_use:
        _aug_kl = aug_kl(meta_metrics["probabilities"], meta_metrics["log_probabilities"], cfg.img_metrics.aug_kld)        
        pred["aug_kl"] = _aug_kl

    if "mink" in cfg.img_metrics.metrics_to_use:
        _min_k = min_k(meta_metrics["all_prob"], cfg.img_metrics.mink)
        pred["mink"] = _min_k

    if "mod_renyi_1_entro" in cfg.img_metrics.metrics_to_use:
        _mod_entropy = mod_entropy(meta_metrics["modified_entropies"])
        pred["mod_renyi_1_entro"] = _mod_entropy

    if "mod_renyi_05_entro" in cfg.img_metrics.metrics_to_use:
        _mod_renyi_05 = mod_renyi(meta_metrics["modified_entropies_alpha_05"])
        pred["mod_renyi_05_entro"] = _mod_renyi_05

    if "mod_renyi_2_entro" in cfg.img_metrics.metrics_to_use:
        _mod_renyi_2 = mod_renyi(meta_metrics["modified_entropies_alpha_2"])
        pred["mod_renyi_2_entro"] = _mod_renyi_2

    if "max_prob_gap" in cfg.img_metrics.metrics_to_use:
        _max_prob_gap = max_prob_gap(meta_metrics["gap_probs"])
        pred["max_prob_gap"] = _max_prob_gap

    if "max_k_renyi_1_entro" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_1_entro = max_entropy(meta_metrics["entropies"], cfg.img_metrics.max_k_renyi_1_entro)
        pred["max_k_renyi_1_entro"] = _max_k_renyi_1_entro

    if "max_k_renyi_2_entro" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_2_entro = max_entropy(meta_metrics["renyi_2_entro"], cfg.img_metrics.max_k_renyi_2_entro)
        pred["max_k_renyi_2_entro"] = _max_k_renyi_2_entro

    if "max_k_renyi_05_entro" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_05_entro = max_entropy(meta_metrics["renyi_05_entro"], cfg.img_metrics.max_k_renyi_05_entro)
        pred["max_k_renyi_05_entro"] = _max_k_renyi_05_entro

    # if "max_k_renyi_inf" in cfg.img_metrics.metrics_to_use:
    #     _max_k_renyi_inf = max_entropy(meta_metrics["renyi_inf_entro"], cfg.img_metrics.max_k_renyi_inf_entro)
    #     pred["max_k_renyi_inf"] = _max_k_renyi_inf

    if "min_k_renyi_1_entro" in cfg.img_metrics.metrics_to_use:
        _min_k_renyi_1_entro = min_entropy(meta_metrics["entropies"], cfg.img_metrics.min_k_renyi_1_entro)
        pred["min_k_renyi_1_entro"] = _min_k_renyi_1_entro

    if "min_k_renyi_2_entro" in cfg.img_metrics.metrics_to_use:
        _min_k_renyi_2_entro = min_entropy(meta_metrics["renyi_2_entro"], cfg.img_metrics.min_k_renyi_2_entro)
        pred["min_k_renyi_2_entro"] = _min_k_renyi_2_entro

    if "min_k_renyi_05_entro" in cfg.img_metrics.metrics_to_use:
        _min_k_renyi_05_entro = min_entropy(meta_metrics["renyi_05_entro"], cfg.img_metrics.min_k_renyi_05_entro)
        pred["min_k_renyi_05_entro"] = _min_k_renyi_05_entro

    # Add our metrics here
    if "cross_entropy_mink" in cfg.img_metrics.metrics_to_use:
        _cross_entropy_mink = cross_entropy_mink(meta_metrics["per_token_CE_loss"], cfg.img_metrics.cross_entropy_mink)
        pred["cross_entropy_min_k"] = _cross_entropy_mink

    if "cross_entropy_diff_mink" in cfg.img_metrics.metrics_to_use:
        _cross_entropy_diff_min_k = cross_entropy_diff_mink(meta_metrics["per_token_CE_loss"], cfg.img_metrics.cross_entropy_mink)
        pred["cross_entropy_diff_min_k"] = _cross_entropy_diff_min_k
        
    if "max_k_no_norn_kl_div" in cfg.img_metrics.metrics_to_use:
        pred["max_k_no_norn_kl_div"], meta["max_k_no_norn_kl_div_tkn_vals"] = renyi_kl_div_maxk(meta_metrics['no_norm_probs'], cfg.img_metrics.max_k_no_norn_kl_div)
        
    if "max_k_renyi_05_kl_div" in cfg.img_metrics.metrics_to_use:
        pred["max_k_renyi_05_kl_div"], meta["max_k_renyi_05_kl_div_tkn_vals"] = renyi_kl_div_maxk(meta_metrics['renyi_05_probs'], cfg.img_metrics.max_k_renyi_05_kl_div)
        
    if "max_k_renyi_1_kl_div" in cfg.img_metrics.metrics_to_use:
        pred["max_k_renyi_1_kl_div"],  meta["max_k_renyi_1_kl_div_tkn_vals"] = renyi_kl_div_maxk(meta_metrics['renyi_1_probs'], cfg.img_metrics.max_k_renyi_1_kl_div)
        
    if "max_k_renyi_2_kl_div" in cfg.img_metrics.metrics_to_use:
        pred["max_k_renyi_2_kl_div"],  meta["max_k_renyi_2_kl_div_tkn_vals"] = renyi_kl_div_maxk(meta_metrics['renyi_2_probs'], cfg.img_metrics.max_k_renyi_2_kl_div)
        
    if "max_k_renyi_inf_kl_div" in cfg.img_metrics.metrics_to_use:
        pred["max_k_renyi_inf_kl_div"],  meta["max_k_renyi_inf_kl_div_tkn_vals"] = renyi_kl_div_maxk(meta_metrics['renyi_inf_probs'], cfg.img_metrics.max_k_renyi_inf_kl_div)
        
    if "max_k_renyi_divergence_025" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_divergence_025 = renyi_divergence_maxk(meta_metrics['probabilities'], cfg.img_metrics.max_k_renyi_divergence_025)
        pred["max_k_renyi_divergence_025"], meta["max_k_renyi_divergence_025_tkn_vals"] = _max_k_renyi_divergence_025
        
    if "max_k_renyi_divergence_05" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_divergence_05 = renyi_divergence_maxk(meta_metrics['probabilities'], cfg.img_metrics.max_k_renyi_divergence_05)
        pred["max_k_renyi_divergence_05"], meta["max_k_renyi_divergence_05_tkn_vals"] = _max_k_renyi_divergence_05
         
    if "max_k_renyi_divergence_2" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_divergence_2 = renyi_divergence_maxk(meta_metrics['probabilities'], cfg.img_metrics.max_k_renyi_divergence_2)
        pred["max_k_renyi_divergence_2"], meta["max_k_renyi_divergence_2_tkn_vals"] = _max_k_renyi_divergence_2
    
    if "max_k_renyi_divergence_4" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_divergence_4 = renyi_divergence_maxk(meta_metrics['probabilities'], cfg.img_metrics.max_k_renyi_divergence_4)
        pred["max_k_renyi_divergence_4"], meta["max_k_renyi_divergence_4_tkn_vals"] = _max_k_renyi_divergence_4
        
    
    return pred, meta
