import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
import ast
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
                    if _metric == 'aug_kl':
                        auc_val, acc_val, auc_low_vals = auc_acc_low(prediction=_metric_val, answers=used_labels, fpr_thresholds=fpr_threshs)
                        auc_results[_part][_metric_category][_metric] = format_to_json(auc_val)
                        acc_results[_part][_metric_category][_metric] = format_to_json(acc_val)

                        for idx, thresh in enumerate(fpr_threshs):
                            auc_low_results_by_fpr.setdefault(_part, dict()).setdefault(thresh, dict()).setdefault(_metric_category, dict())
                            auc_low_results_by_fpr[_part][thresh][_metric_category][_metric] = format_to_json(auc_low_vals[idx])
                    
                    # Subcase: Metrics with settings (mink ratios)
                    elif _metric in ['mink', 'min_k_renyi_05_entro', 'min_k_renyi_1_entro', 'max_k_renyi_1_entro', 'max_k_renyi_05_entro']:
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


def slice_preds(preds, n):
    """
    Recursively walks a preds-shaped nested dict (as produced by inference()) and slices every
    leaf score list down to its first `n` entries. Used by job_type=tune_and_eval to recover
    target-set-only predictions from a single combined [target-set rows..., reference-set
    rows...] inference pass (target rows are always placed first in that concatenation), so
    the eval output it writes is scored on exactly the same target-set samples/order
    job_type=evaluation would produce, via the same evaluate() call and preds.json shape --
    no second inference pass needed.
    """
    if isinstance(preds, dict):
        return {k: slice_preds(v, n) for k, v in preds.items()}
    if isinstance(preds, list):
        return preds[:n]
    return preds


def compute_tuning_separation(preds, tune_labels, cfg=None):
    """
    Walks `preds` the same way evaluate() does, but scores each leaf (per-sample score list)
    against tune_labels (1 = target-set sample, 0 = reference-set sample) instead of the real
    member/nonmember label. For every leaf (metric/aug/setting), records both:
      - gap = mean(target-set scores) - mean(reference-set scores)   (raw divergence
        separation, i.e. the paper's Delta-D-bar(sigma) but for target-vs-reference)
      - tune_auc = ROC-AUC discriminating target-set vs reference-set samples by that score
        (the signal the existing hyperparam_tuning job_type already computes via evaluate())
    plus the raw target/reference score split itself, for plotting divergence-vs-noise curves.

    Returns (raw_split, separation, best_by_setting):
      raw_split[part][category][metric][...] = {"target": [...], "reference": [...]}
      separation[part][category][metric][...] = {"mean_target":.., "mean_reference":..,
        "gap":.., "tune_auc":..}
        (both mirror preds' own nesting -- kld_metrics/renyi_div_metrics keep the aug/setting
        levels, baseline_metrics keep whatever sub-nesting that metric has, everything else is
        a bare leaf.)
      best_by_setting[part][category][metric][aug][k_ratio_str] = {
          "best_std_by_gap": std, "gap": val, "best_std_by_tune_auc": std, "tune_auc": val
      }
        -- only populated for kld_metrics/renyi_div_metrics settings that parse as a
        {'std':..., 'k_ratio':...} dict (i.e. the actual noise sweep, not 'aggregated' entries).
    """
    # Mirror evaluate()'s test_run truncation (eval.py ~line 84): under test_run, preds only
    # cover the first (batch_size * test_number_of_batches) samples, so tune_labels -- which
    # comes straight from the full untruncated dataset -- must be sliced to match, or the
    # boolean index below fails with a length mismatch (e.g. 5 preds vs 450 labels).
    used_tune_labels = tune_labels
    if cfg is not None and cfg.job_meta_params.test_run:
        used_tune_labels = tune_labels[:(cfg.inference.batch_size * cfg.inference.test_number_of_batches)]
    tune_labels_arr = np.asarray(used_tune_labels, dtype=bool)

    def _leaf(scores):
        scores = np.asarray(scores, dtype=np.float64)
        target_scores = scores[tune_labels_arr]
        reference_scores = scores[~tune_labels_arr]
        mean_t = float(np.mean(target_scores)) if target_scores.size else float('nan')
        mean_r = float(np.mean(reference_scores)) if reference_scores.size else float('nan')
        tune_auc_val = float('nan')
        if target_scores.size and reference_scores.size and np.all(np.isfinite(scores)):
            try:
                _, _, tune_auc_val, _ = sweep(scores, tune_labels_arr)
                tune_auc_val = float(tune_auc_val)
            except Exception:
                tune_auc_val = float('nan')
        raw = {"target": target_scores.tolist(), "reference": reference_scores.tolist()}
        sep = {"mean_target": mean_t, "mean_reference": mean_r, "gap": mean_t - mean_r, "tune_auc": tune_auc_val}
        return raw, sep

    def _parse_setting(setting_name):
        try:
            parsed = ast.literal_eval(setting_name)
        except (ValueError, SyntaxError):
            return None
        return parsed if isinstance(parsed, dict) else None

    raw_split, separation, best_by_setting = dict(), dict(), dict()

    for part, part_pred in preds.items():
        raw_split[part], separation[part], best_by_setting[part] = dict(), dict(), dict()

        for metric_category, metrics in part_pred.items():
            raw_split[part][metric_category] = dict()
            separation[part][metric_category] = dict()
            best_by_setting[part][metric_category] = dict()

            for metric, metric_val in metrics.items():
                if metric_category in {"kld_metrics", "renyi_div_metrics"}:
                    raw_split[part][metric_category][metric] = dict()
                    separation[part][metric_category][metric] = dict()
                    best_by_setting[part][metric_category][metric] = dict()

                    for aug, aug_settings in metric_val.items():
                        raw_split[part][metric_category][metric][aug] = dict()
                        separation[part][metric_category][metric][aug] = dict()

                        by_k = defaultdict(list)  # k_ratio -> [(std, gap, tune_auc), ...]
                        for setting, scores in aug_settings.items():
                            raw, sep = _leaf(scores)
                            raw_split[part][metric_category][metric][aug][setting] = raw
                            separation[part][metric_category][metric][aug][setting] = sep

                            setting_dict = _parse_setting(setting)
                            if setting_dict is not None and 'std' in setting_dict and 'k_ratio' in setting_dict:
                                by_k[setting_dict['k_ratio']].append(
                                    (float(setting_dict['std']), sep['gap'], sep['tune_auc'])
                                )

                        k_result = dict()
                        for k_ratio, points in by_k.items():
                            finite_gap = [p for p in points if np.isfinite(p[1])]
                            finite_auc = [p for p in points if np.isfinite(p[2])]
                            entry = dict()
                            if finite_gap:
                                best_std, best_gap, _ = max(finite_gap, key=lambda p: abs(p[1]))
                                entry["best_std_by_gap"] = best_std
                                entry["gap"] = best_gap
                            if finite_auc:
                                best_std, _, best_auc = max(finite_auc, key=lambda p: p[2])
                                entry["best_std_by_tune_auc"] = best_std
                                entry["tune_auc"] = best_auc
                            if entry:
                                k_result[str(k_ratio)] = entry
                        if k_result:
                            best_by_setting[part][metric_category][metric][aug] = k_result

                elif metric_category == "baseline_metrics":
                    if metric == 'aug_kl':
                        raw, sep = _leaf(metric_val)
                        raw_split[part][metric_category][metric] = raw
                        separation[part][metric_category][metric] = sep
                    elif metric in ['mink', 'min_k_renyi_05_entro', 'min_k_renyi_1_entro',
                                     'max_k_renyi_1_entro', 'max_k_renyi_05_entro']:
                        raw_split[part][metric_category][metric] = dict()
                        separation[part][metric_category][metric] = dict()
                        for setting, scores in metric_val.items():
                            raw, sep = _leaf(scores)
                            raw_split[part][metric_category][metric][setting] = raw
                            separation[part][metric_category][metric][setting] = sep
                    # else: unrecognized baseline metric -- this is a diagnostics helper, not
                    # the scoring path of record, so skip rather than raise.

                else:
                    raw, sep = _leaf(metric_val)
                    raw_split[part][metric_category][metric] = raw
                    separation[part][metric_category][metric] = sep

    return raw_split, separation, best_by_setting


















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

