"""
Verifies the vectorized get_meta_metrics_by_part (src/metrics/meta_metrics.py) produces
numerically identical results to the original per-token-loop implementation it replaced.

Motivation: profiling the flickr/LLaVA mixture sweep found get_meta_metrics_by_part costing
~16.5s/batch, almost entirely from a double Python loop (16 samples x ~576 image tokens) where
each iteration called .item()/.cpu(), forcing a GPU sync every single time (~18,400 syncs/batch).
The fix hoists the token-axis-only computations (entropy, renyi_1, no_norm, renyi_inf) out of the
per-token loop into one vectorized op per *sample* -- keeping the per-sample loop intact, since
sequence lengths genuinely aren't always uniform across a batch (confirmed empirically: 1/600
samples in run_1's real data had 577 image tokens instead of 576) -- so this test specifically
checks a ragged-length batch, not just a uniform one.

`get_meta_metrics_by_part_REFERENCE` below is byte-for-byte the pre-fix implementation (see git
history on the MergeTuning branch, commit f941264, src/metrics/meta_metrics.py), kept here as a
frozen reference rather than re-derived from git at test time.

Run with: PYTHONPATH=<repo>/mia python mia/tests/test_meta_metrics_vectorized.py
(deliberately tiny synthetic dimensions -- 24 tokens x 500 vocab, not production's 576 x 32000 --
since this checks correctness, not performance, and the original per-token loop is slow enough at
full scale that running this on the login node/CI as a bare process is worth avoiding)
"""
import os
import sys

import torch
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics.meta_metrics import get_meta_metrics_by_part as get_meta_metrics_by_part_NEW


def renyi_probs_REFERENCE(token_probs_clamped, alpha):
    if alpha == "inf":
        max_values, max_indices = torch.max(token_probs_clamped, dim=-1, keepdim=True)
        renyi_normalized = torch.zeros_like(token_probs_clamped)
        renyi_normalized.scatter_(-1, max_indices, 1.0)
        return renyi_normalized
    else:
        renyi_numerator = torch.pow(token_probs_clamped, alpha)
        renyi_denominator = torch.sum(renyi_numerator, dim=-1, keepdim=True)
        return renyi_numerator / renyi_denominator


def get_meta_metrics_by_part_REFERENCE(total_parts, part, cfg):
    """Frozen copy of the pre-fix per-token-loop implementation (MergeTuning@f941264), trimmed to
    just the metrics this test actually exercises (entropies/renyi_1/no_norm/renyi_inf) -- the
    other conditional blocks (renyi_05/2, mink, modified_entropies, per_token_CE_loss) are
    untouched by the perf fix and not re-verified here."""
    meta_metrics = {name: dict() for name in ["entropies", "no_norm_probs", "renyi_1_probs",
                                               "renyi_inf_probs", "gap_probs", "max_probs"]}
    epsilon = 1e-4

    for aug_type, aug_results in total_parts.items():
        for name in meta_metrics:
            meta_metrics[name][aug_type] = [[] for _ in range(len(aug_results))]

        for aug_idx, aug_result in enumerate(aug_results):
            n_samples = len(aug_result[part]["input_ids"])
            for name in meta_metrics:
                meta_metrics[name][aug_type][aug_idx] = [[] for _ in range(n_samples)]

            for _batch_idx in range(n_samples):
                for _token_idx, token_id in enumerate(aug_result[part]["input_ids"][_batch_idx][1:]):
                    token_probs = aug_result[part]["probabilities"][_batch_idx][_token_idx, :]
                    token_log_probs = aug_result[part]["log_probabilities"][_batch_idx][_token_idx, :]
                    token_probs_clamped = torch.clamp(token_probs, min=epsilon, max=1 - epsilon)
                    token_log_probs_clamped = token_probs_clamped.log()

                    entropy = -(token_probs * token_log_probs_clamped).sum().item()
                    meta_metrics["entropies"][aug_type][aug_idx][_batch_idx].append(entropy)
                    meta_metrics["renyi_1_probs"][aug_type][aug_idx][_batch_idx].append(
                        renyi_probs_REFERENCE(token_probs_clamped, 1).detach().cpu())

                    if (
                        "max_k_no_norn_kl_div" in cfg.img_metrics.metrics_to_use
                        or "max_k_no_norn_kl_div_tkn_vals" in cfg.img_metrics.get_proc_meta_metrics
                        or "no_norm_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_k_no_norn_kl_div_ripple" in cfg.img_metrics.metrics_to_use
                        or "max_k_no_norn_kl_div_tkn_vals_ripple" in cfg.img_metrics.get_proc_meta_metrics
                    ):
                        meta_metrics["no_norm_probs"][aug_type][aug_idx][_batch_idx].append(
                            token_probs_clamped.detach().cpu())

                    if (
                        "max_k_renyi_inf_kl_div" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_inf_kl_div_tkn_vals" in cfg.img_metrics.get_proc_meta_metrics
                        or "gap_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "renyi_inf_probs" in cfg.img_metrics.get_raw_meta_metrics
                        or "max_k_renyi_inf_kl_div_ripple" in cfg.img_metrics.metrics_to_use
                        or "max_k_renyi_inf_kl_div_tkn_vals_ripple" in cfg.img_metrics.get_proc_meta_metrics
                    ):
                        max_p = token_log_probs_clamped.max().item()
                        second_p = token_log_probs_clamped[token_log_probs_clamped != token_log_probs_clamped.max()].max().item()
                        meta_metrics["gap_probs"][aug_type][aug_idx][_batch_idx].append(max_p - second_p)
                        meta_metrics["max_probs"][aug_type][aug_idx][_batch_idx].append(max_p)
                        meta_metrics["renyi_inf_probs"][aug_type][aug_idx][_batch_idx].append(
                            renyi_probs_REFERENCE(token_probs_clamped, "inf").detach().cpu())

    return meta_metrics


torch.manual_seed(0)

BATCH_SIZE = 16
# Deliberately much smaller than production (576 tokens x 32000 vocab): correctness doesn't
# depend on matching production's exact dimensions, only on exercising the same code paths
# (multi-token sequences, ragged lengths, multi-dim reductions).
SEQ_LEN = 24
VOCAB_SIZE = 500

cfg = OmegaConf.create({
    "img_metrics": {
        "metrics_to_use": ["max_k_no_norn_kl_div", "max_k_renyi_inf_kl_div"],
        "get_proc_meta_metrics": [],
        "get_raw_meta_metrics": [],
    }
})


def make_synthetic_aug_result(seq_len=SEQ_LEN, one_sample_shorter=False):
    """`one_sample_shorter` makes ONE sample in the batch one token shorter than the rest,
    mirroring the real ragged-length edge case (1/600 samples in run_1's actual data had a 577
    vs 576 img token count) -- the per-sample loop must handle this correctly."""
    probs_list, log_probs_list, input_ids_list = [], [], []
    for i in range(BATCH_SIZE):
        this_len = seq_len - 1 if (one_sample_shorter and i == 3) else seq_len
        logits = torch.randn(this_len, VOCAB_SIZE) * 3.0  # spread out, not near-uniform
        probs_list.append(torch.softmax(logits, dim=-1))
        log_probs_list.append(torch.log_softmax(logits, dim=-1))
        input_ids_list.append([0] + torch.randint(0, VOCAB_SIZE, (this_len,)).tolist())
    return {"img": {"probabilities": probs_list, "log_probabilities": log_probs_list, "input_ids": input_ids_list}}


def compare(old_val, new_val, path):
    if isinstance(old_val, torch.Tensor) or isinstance(new_val, torch.Tensor):
        old_t, new_t = torch.as_tensor(old_val), torch.as_tensor(new_val)
        if old_t.shape != new_t.shape:
            return f"SHAPE MISMATCH at {path}: {old_t.shape} vs {new_t.shape}"
        max_diff = (old_t.float() - new_t.float()).abs().max().item()
        return None if max_diff <= 1e-4 else f"VALUE MISMATCH at {path}: max_abs_diff={max_diff}"
    elif isinstance(old_val, (int, float, np.floating)):
        return None if abs(old_val - new_val) <= 1e-4 else f"VALUE MISMATCH at {path}: {old_val} vs {new_val}"
    elif isinstance(old_val, list):
        for i, (o, n) in enumerate(zip(old_val, new_val)):
            err = compare(o, n, f"{path}[{i}]")
            if err:
                return err
        return None
    return None


def run_test(one_sample_shorter, label):
    print(f"\n=== {label} ===")
    aug_result = make_synthetic_aug_result(one_sample_shorter=one_sample_shorter)
    total_parts = {"orig": [aug_result]}

    old = get_meta_metrics_by_part_REFERENCE(total_parts, "img", cfg)
    new = get_meta_metrics_by_part_NEW(total_parts, "img", cfg)

    errors = []
    for metric_name in ["entropies", "renyi_1_probs", "no_norm_probs", "gap_probs", "max_probs", "renyi_inf_probs"]:
        err = compare(old[metric_name]["orig"][0], new[metric_name]["orig"][0], metric_name)
        (errors.append(err) if err else print(f"  {metric_name}: MATCH"))

    if errors:
        print("FAILURES:")
        for e in errors:
            print(" ", e)
        return False
    print(f"{label}: ALL METRICS MATCH")
    return True


if __name__ == "__main__":
    ok1 = run_test(one_sample_shorter=False, label="uniform-length batch (typical case)")
    ok2 = run_test(one_sample_shorter=True, label="ragged-length batch (1 sample shorter, matches the real ~0.17% edge case)")
    print("\n=== OVERALL:", "PASS" if (ok1 and ok2) else "FAIL", "===")
    sys.exit(0 if (ok1 and ok2) else 1)
