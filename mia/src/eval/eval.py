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

    
    
    
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def auc_acc_low(prediction, answers, sweep_fn=sweep, low=0.05 ):
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<low)[0][-1]]

    return auc, acc, low


def evaluate(preds, labels, part, cfg):
    """
    pred: predictions with every metrics
    """

    auc_results = dict()
    acc_results = dict()
    auc_low_results = dict()
    
    used_labels = labels
    
    if cfg.job_meta_params.test_run:
        used_labels = labels[:(cfg.inference.batch_size* cfg.inference.test_number_of_batches)]
        

    for _part, _part_pred in preds.items():
        if _part not in auc_results:
            auc_results[_part] = dict()
            acc_results[_part] = dict()
            auc_low_results[_part] = dict()
        for _metric, _metric_val in _part_pred.items():
            if isinstance(_metric_val, list):
                
                auc_val, acc_val, auc_low_val = auc_acc_low(prediction=_metric_val, answers=used_labels)
                auc_results[_part][_metric] = auc_val
                acc_results[_part][_metric] = acc_val
                auc_low_results[_part][_metric] = auc_low_val

            else:
                auc_results[_part][_metric] = dict()
                acc_results[_part][_metric] = dict()
                auc_low_results[_part][_metric] = dict()
                for _sub_metric, _sub_metric_val in _metric_val.items():
                    
                    auc_val, acc_val, auc_low_val = auc_acc_low(prediction=_sub_metric_val, answers=used_labels)
                    auc_results[_part][_metric][_sub_metric] = format_to_json(auc_val)
                    acc_results[_part][_metric][_sub_metric] = format_to_json(acc_val)
                    auc_low_results[_part][_metric][_sub_metric] = format_to_json(auc_low_val)

    return auc_results, acc_results, auc_low_results

