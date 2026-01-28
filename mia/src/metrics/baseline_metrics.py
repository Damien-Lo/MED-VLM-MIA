import statistics
import numpy as np
from src.metrics.utils import kl_divergence

"""
Baseline img metrics computation functions are listed here
"""

def aug_kl(probabilities, log_probabilities, cfg):

    result = list()
    aug_names = list(probabilities.keys())
    if len(aug_names) == 1:
        return None
    
    for _sample_idx, (_orig_prob, _orig_log_prob) in enumerate(zip(probabilities["orig"][0], log_probabilities["orig"][0])):
        # iterating over samples
        _kld = list()
        for aug_name in aug_names:
            if aug_name == "orig":
                continue
            _kld.append(kl_divergence(_orig_prob,
                                      _orig_log_prob,
                                      probabilities[aug_name][0][_sample_idx]))
                                     # change [0] to something else or add a new logic if we use multiple params for each augmentation.
        result.append(-statistics.mean(_kld).item())
    return result


def min_k(all_prob, cfg):
    
    ratio = cfg.ratio     

    result = dict()
    for _ratio in ratio: 
        _key = f"Min_{_ratio*100}% prob"
        result[_key] = list()
        for _prob in all_prob["orig"][0]:
            k_length = int(len(_prob)*_ratio)
            if k_length == 0:
                k_length = 1
            topk_prob = np.sort(_prob)[:k_length]
            result[_key].append(-1 * np.mean(topk_prob).item())

    return result

def mod_entropy(mod_entropy):
    
    result = list()
    for _sample in mod_entropy["orig"][0]:
        result.append(np.nanmean(_sample).item())

    return result

def mod_renyi(renyi):

    result = list()
    for _sample in renyi["orig"][0]:
        result.append(np.nanmean(_sample).item())
    
    return result

def max_prob_gap(gap_p):

    result = list()
    for _sample in gap_p["orig"][0]:
        result.append(-np.mean(_sample).item())
    
    return result

def max_entropy(entropy, cfg):
    
    ratio = cfg.ratio
    result = dict()

    for _ratio in ratio: 
        _key = f"Max_{_ratio*100}% "+cfg.suffix
        result[_key] = list()
        for _entro in entropy["orig"][0]:
            k_length = int(len(_entro)*_ratio)
            if k_length == 0:
                k_length = 1
            topk_prob = np.sort(_entro)[-k_length:]
            result[_key].append(np.mean(topk_prob).item())
    
    return result

def min_entropy(renyi, cfg):
    
    ratio  = cfg.ratio
    result = dict()

    for _ratio in ratio:
        _key = f"Min_{_ratio*100}% "+cfg.suffix
        result[_key] = list()
        for _renyi in renyi["orig"][0]:
            k_length = int(len(_renyi)*_ratio)
            if k_length == 0:
                k_length = 1
            topk_prob = np.sort(_renyi)[:k_length]
            result[_key].append(np.mean(topk_prob).item())
    
    return result

