import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random
import os
import sys


def format_to_json(value):
        if isinstance(value, float):
            return value
        if isinstance(value, (np.integer, np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.ndarray,)):
            return value.tolist()
        return super().default(value)


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    
    
    score = np.asarray(score, dtype=np.float64)
    x = np.asarray(x, dtype=bool)

    # Diagnose non-finite scores early (this is what sklearn is complaining about)
    finite_mask = np.isfinite(score)
    if not np.all(finite_mask):
        bad_idx = np.where(~finite_mask)[0]
        print(f"[sweep] Non-finite score detected: {bad_idx.size} / {score.size}")
        print(f"  First bad indices: {bad_idx[:10].tolist()}")
        print(f"  Bad values (first 10): {score[bad_idx[:10]]}")
        # Option A: raise to stop (recommended during debugging)
        raise ValueError("score contains inf or NaN; cannot compute ROC.")
        # Option B (if you *must* continue): filter them out
        # score = score[finite_mask]
        # x = x[finite_mask]

    # Also useful: check label validity
    uniq = np.unique(x)
    if uniq.size != 2:
        raise ValueError(f"x must be binary with both classes present; got unique={uniq}")

    
    fpr, tpr, _ = roc_curve(x, score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def auc_acc_low(prediction, answers, sweep_fn=sweep, fpr_thresholds=[0.05]):
    fpr, tpr, auc, acc = sweep_fn(
        np.asarray(prediction),
        np.asarray(answers, dtype=bool)
    )

    tpr_low = list()
    for thresh in fpr_thresholds:
        mask = fpr <= thresh
        tpr_low.append(np.max(tpr[mask]) if np.any(mask) else 0.0)

    return auc, acc, tpr_low



def evaluate(preds, labels, part, cfg):
    fpr_threshs = cfg.inference.fpr_threshs if cfg.inference.fpr_threshs is not None else [0.05]

    auc_results = dict()
    acc_results = dict()
    auc_low_results_by_fpr = dict()

    used_labels = labels
    if cfg.job_meta_params.test_run:
        used_labels = labels[:(cfg.inference.batch_size * cfg.inference.test_number_of_batches)]

    for _part, _part_pred in preds.items():
        auc_results.setdefault(_part, dict())
        acc_results.setdefault(_part, dict())

        for _metric_category, _metrics in _part_pred.items():
            auc_results[_part].setdefault(_metric_category, dict())
            acc_results[_part].setdefault(_metric_category, dict())

            for _metric, _metric_val in _metrics.items():
                
                # --- CASE 1: Metrics with Augmentation Settings (KLD / Renyi) ---
                if _metric_category in {"kld_metrics", "renyi_div_metrics"}:
                    auc_results[_part][_metric_category].setdefault(_metric, dict())
                    acc_results[_part][_metric_category].setdefault(_metric, dict())

                    for aug, aug_settings in _metric_val.items():
                        auc_results[_part][_metric_category][_metric].setdefault(aug, dict())
                        acc_results[_part][_metric_category][_metric].setdefault(aug, dict())

                        for aug_setting, scores in aug_settings.items():
                            auc_val, acc_val, auc_low_vals = auc_acc_low(
                                prediction=scores,
                                answers=used_labels,
                                fpr_thresholds=fpr_threshs,
                            )
                            auc_results[_part][_metric_category][_metric][aug][aug_setting] = format_to_json(auc_val)
                            acc_results[_part][_metric_category][_metric][aug][aug_setting] = format_to_json(acc_val)

                            for idx, thresh in enumerate(fpr_threshs):
                                auc_low_results_by_fpr.setdefault(_part, dict()).setdefault(thresh, dict()).setdefault(_metric_category, dict()).setdefault(_metric, dict()).setdefault(aug, dict())
                                auc_low_results_by_fpr[_part][thresh][_metric_category][_metric][aug][aug_setting] = format_to_json(auc_low_vals[idx])

                # --- CASE 2: Baseline Metrics (mink, aug_kl, etc.) ---
                elif _metric_category == 'baseline_metrics':
                    # Subcase: Simple baseline (no sub-settings)
                    if _metric in ('aug_kl', 'vision_max_prob_gap'):
                        auc_val, acc_val, auc_low_vals = auc_acc_low(prediction=_metric_val, answers=used_labels, fpr_thresholds=fpr_threshs)
                        auc_results[_part][_metric_category][_metric] = format_to_json(auc_val)
                        acc_results[_part][_metric_category][_metric] = format_to_json(acc_val)

                        for idx, thresh in enumerate(fpr_threshs):
                            auc_low_results_by_fpr.setdefault(_part, dict()).setdefault(thresh, dict()).setdefault(_metric_category, dict())
                            auc_low_results_by_fpr[_part][thresh][_metric_category][_metric] = format_to_json(auc_low_vals[idx])

                    # Subcase: Metrics with settings (mink ratios)
                    elif _metric in ['mink', 'min_k_renyi_05_entro', 'min_k_renyi_1_entro', 'max_k_renyi_1_entro', 'max_k_renyi_05_entro',
                                      'vision_max_k_renyi_1_entro', 'vision_max_k_renyi_2_entro', 'vision_max_k_renyi_05_entro',
                                      'vision_min_k_renyi_1_entro', 'vision_min_k_renyi_2_entro', 'vision_min_k_renyi_05_entro']:
                        auc_results[_part][_metric_category].setdefault(_metric, dict())
                        acc_results[_part][_metric_category].setdefault(_metric, dict())
                        
                        for setting, scores in _metric_val.items():
                            auc_val, acc_val, auc_low_vals = auc_acc_low(prediction=scores, answers=used_labels, fpr_thresholds=fpr_threshs)
                            
                            auc_results[_part][_metric_category][_metric][setting] = format_to_json(auc_val)
                            acc_results[_part][_metric_category][_metric][setting] = format_to_json(acc_val)

                            for idx, thresh in enumerate(fpr_threshs):
                                auc_low_results_by_fpr.setdefault(_part, dict()).setdefault(thresh, dict()).setdefault(_metric_category, dict()).setdefault(_metric, dict())
                                auc_low_results_by_fpr[_part][thresh][_metric_category][_metric][setting] = format_to_json(auc_low_vals[idx])
                    else:
                        # Fallback for unexpected baseline metrics
                        raise ValueError(f"Unknown baseline metric: {_metric}")

                # --- CASE 3: Metrics with no settings (loss, entropy, etc.) ---
                else:
                    auc_val, acc_val, auc_low_vals = auc_acc_low(prediction=_metric_val, answers=used_labels, fpr_thresholds=fpr_threshs)
                    auc_results[_part][_metric_category][_metric] = format_to_json(auc_val)
                    acc_results[_part][_metric_category][_metric] = format_to_json(acc_val)

                    for idx, thresh in enumerate(fpr_threshs):
                        auc_low_results_by_fpr.setdefault(_part, dict()).setdefault(thresh, dict()).setdefault(_metric_category, dict())
                        auc_low_results_by_fpr[_part][thresh][_metric_category][_metric] = format_to_json(auc_low_vals[idx])

    return auc_results, acc_results, auc_low_results_by_fpr



















# def evaluate(preds, labels, part, cfg):
#     """
#     pred: predictions with every metrics
#     """
    
#     fpr_threshs = (
#         cfg.inference.fpr_threshs
#         if cfg.inference.fpr_threshs is not None
#         else [0.05]
#     )

#     auc_results = dict()
#     acc_results = dict()
#     auc_low_results = dict()
    
#     used_labels = labels
    
#     if cfg.job_meta_params.test_run:
#         used_labels = labels[:(cfg.inference.batch_size* cfg.inference.test_number_of_batches)]
        
        
#     for _part, _part_pred in preds.items():
#         if _part not in auc_results:
#             auc_results[_part] = dict()
#             acc_results[_part] = dict()
#         for _metric_category, _metrics in _part_pred.items():
#             if _metric_category not in auc_results[_part]:
#                 auc_results[_part][_metric_category] = dict()
#                 acc_results[_part][_metric_category] = dict()
                
#             for _metric, _metric_val in _metrics.items(): 
#                 if _metric not in auc_results[_part][_metric_category]:
#                     auc_results[_part][_metric_category][_metric] = dict()
#                     acc_results[_part][_metric_category][_metric] = dict()
#                 # Metrics that have settings    
#                 if _metric_category in set(['kld_metrics', 'renyi_div_metrics']):
#                     for aug, aug_settings in _metric_val.items():
#                         if aug not in auc_results[_part][_metric_category][_metric]:
#                             auc_results[_part][_metric_category][_metric][aug] = dict()
#                             acc_results[_part][_metric_category][_metric][aug] = dict()
#                         for aug_setting, scores in aug_settings.items():
#                             auc_val, acc_val, auc_low_vals = auc_acc_low(prediction=scores, answers=used_labels, fpr_thresholds=fpr_threshs)
#                             auc_results[_part][_metric_category][_metric][aug][aug_setting] = format_to_json(auc_val)
#                             acc_results[_part][_metric_category][_metric][aug][aug_setting] = format_to_json(acc_val)
                            
#                             for idx, thresh in enumerate(fpr_threshs):
#                                 auc_low_results_by_fpr.setdefault(fpr_key, {}).setdefault(_part, {}).setdefault(_metric_category, {}).setdefault(_metric, {}).setdefault(aug, {})
#                                 auc_low_results[thresh][_part][_metric_category][_metric][aug][setting_key] = format_to_json(auc_low_vals[idx])
                            
#                 # Metrics that have no settings / submetrics               
#                 else:
#                     auc_val, acc_val, auc_low_vals = auc_acc_low(prediction=_metric_val, answers=used_labels, fpr_thresholds=fpr_threshs)
#                     auc_results[_part][_metric_category][_metric] = auc_val
#                     acc_results[_part][_metric_category][_metric] = acc_val
#                     auc_low_results[_part][_metric_category][_metric] = auc_low_val
            
            
            
#             # for _metric, _metric_val in _metrics.items():
#             #     if isinstance(_metric_val, list):
#             #         auc_val, acc_val, auc_low_val = auc_acc_low(prediction=_metric_val, answers=used_labels)
#             #         auc_results[_part][_metric] = auc_val
#             #         acc_results[_part][_metric] = acc_val
#             #         auc_low_results[_part][_metric] = auc_low_val

#             #     else:
#             #         auc_results[_part][_metric] = dict()
#             #         acc_results[_part][_metric] = dict()
#             #         auc_low_results[_part][_metric] = dict()
#             #         for _sub_metric, _sub_metric_val in _metric_val.items():
#             #             auc_val, acc_val, auc_low_val = auc_acc_low(prediction=_sub_metric_val, answers=used_labels)
#             #             auc_results[_part][_metric][_sub_metric] = format_to_json(auc_val)
#             #             acc_results[_part][_metric][_sub_metric] = format_to_json(acc_val)
#             #             auc_low_results[_part][_metric][_sub_metric] = format_to_json(auc_low_val)

#     return auc_results, acc_results, auc_low_results

