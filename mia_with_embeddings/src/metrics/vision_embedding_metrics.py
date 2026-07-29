"""
Vision-encoder-embedding variants of the KL/Renyi max-k metrics (proposed_metrics.py)
and the distributional baseline metrics (baseline_metrics.py).

These operate on the raw, pre-projection vision-encoder embeddings captured via
HulumedQwen2ForCausalLM.get_last_vision_encoder_embeddings() (see
src/model/infer.py::mod_infer_batch_hulu), instead of the LLM's next-token
probability distributions.

A vision-encoder patch embedding is a continuous vector, not a probability
distribution, so to reuse the existing KL/Renyi divergence math each patch's
embedding is softmaxed over its hidden dimension (treated as an unnormalized
logit row over a pseudo-vocab of size hidden_dim). The resulting pseudo-probability
row is fed into the same per-token divergence formulas already used for text
tokens (kl_div_per_token / renyi_div_per_token), imported unmodified from
proposed_metrics.py.

Every metric here is produced in two variants:
  - "vision"            : divergence/entropy computed on vision-token embeddings alone.
  - "vision_img_concat" : the vision-token per-token score array concatenated with the
                          existing "img" part's per-token score array (computed from the
                          LLM logits at the <image> placeholder positions) before the
                          max/min-k step. Only produced if "img" is one of cfg.img_metrics.parts.

This module is purely additive: it does not modify baseline_metrics.py,
proposed_metrics.py, meta_metrics.py, or img_metrics.py. Its output dicts use the
same "kld_metrics" / "renyi_div_metrics" / "baseline_metrics" shape that
src/eval/eval.py::evaluate already knows how to walk, as long as the caller merges
the returned "vision" / "vision_img_concat" dicts into the same top-level `preds`
dict under those keys (see src/inference/loop.py).
"""
import ast
import numpy as np
import torch

from src.data.augmentations import get_augmentations
from src.metrics.meta_metrics import renyi_probs
from src.metrics.proposed_metrics import kl_div_per_token, renyi_div_per_token, get_token_slice


# =========================================================================
# Probability helpers
# =========================================================================

def _vision_token_probs(embedding, alpha=None, eps=1e-12):
    """embedding: torch.Tensor [num_vision_tokens, hidden_dim] (raw vision-encoder
    output) -> numpy [num_vision_tokens, hidden_dim] pseudo-probabilities."""
    probs = torch.nn.functional.softmax(embedding.float(), dim=-1)
    probs = torch.clamp(probs, min=eps, max=1 - eps)
    if alpha is not None:
        probs = renyi_probs(probs, alpha)
    return probs.cpu().numpy()


def _img_token_probs(prob_tensor, alpha=None, eps=1e-12):
    """prob_tensor: already-softmaxed LLM probabilities at <image> positions,
    [num_img_tokens, vocab] (target_parts[...]["img"]["probabilities"][sample_idx])."""
    probs = torch.clamp(prob_tensor.float(), min=eps, max=1 - eps)
    if alpha is not None:
        probs = renyi_probs(probs, alpha)
    return probs.cpu().numpy()


def _orig_img_probs_list(target_parts):
    return target_parts["orig"][0]["img"]["probabilities"]


def _aug_img_probs_list(target_parts, aug, setting_idx):
    return target_parts[aug][setting_idx]["img"]["probabilities"]


def _entropy_per_token(probs, alpha):
    """probs: numpy [num_tokens, dim], already clamped to (eps, 1-eps)."""
    if alpha == 1:
        return -(probs * np.log(probs)).sum(axis=1)
    return (1 / (1 - alpha)) * np.log(np.sum(probs ** alpha, axis=1))


def _gap_per_token(probs):
    """probs: numpy [num_tokens, dim], already clamped to (eps, 1-eps)."""
    log_probs = np.log(probs)
    sorted_logs = np.sort(log_probs, axis=1)
    return sorted_logs[:, -1] - sorted_logs[:, -2]


# =========================================================================
# KL-divergence max-k family (mirrors proposed_metrics.renyi_kl_div_maxk /
# renyi_kl_div_ripple_maxk)
# =========================================================================

def _kl_div_maxk_engine(vision_embeddings, target_parts, metric_cfg, cfg,
                         alpha=None, concat_with_img=True, ripple=False, eps=1e-12):
    result = dict()
    meta = dict()

    ratio = metric_cfg.ratio
    _, aug_desc_dict = get_augmentations(cfg)

    orig_vision_probs = [_vision_token_probs(e, alpha, eps) for e in vision_embeddings["orig"]]
    orig_vision_logs = [np.log(p + eps) for p in orig_vision_probs]

    if concat_with_img:
        orig_img_probs = [_img_token_probs(p, alpha, eps) for p in _orig_img_probs_list(target_parts)]
        orig_img_logs = [np.log(p + eps) for p in orig_img_probs]

    setting_version_accumilator = metric_cfg.augmentation_accumilator
    aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in vision_embeddings.items():
        if aug == "orig":
            continue

        all_settings_in_aug = dict()
        for setting_idx, setting in enumerate(settings):
            key = aug_desc_dict[aug][setting_idx]
            all_samples_in_setting_values = list()
            for sample_idx, aug_emb in enumerate(setting):
                aug_probs = _vision_token_probs(aug_emb, alpha, eps)
                aug_log = np.log(aug_probs + eps)
                kl = kl_div_per_token(orig_vision_probs[sample_idx], orig_vision_logs[sample_idx], aug_log)

                if concat_with_img:
                    aug_img_probs = _img_token_probs(_aug_img_probs_list(target_parts, aug, setting_idx)[sample_idx], alpha, eps)
                    aug_img_log = np.log(aug_img_probs + eps)
                    img_kl = kl_div_per_token(orig_img_probs[sample_idx], orig_img_logs[sample_idx], aug_img_log)
                    kl = np.concatenate([kl, img_kl])

                all_samples_in_setting_values.append(kl)

            all_settings_in_aug[str(key)] = all_samples_in_setting_values

        meta[aug] = all_settings_in_aug

        if aug not in result:
            result[aug] = dict()

        if 'none' in setting_version_accumilator:
            _fill_none_accumulated(result[aug], all_settings_in_aug, ratio, ripple, cfg)

        _accumulate_across_settings(aug_aggregated_per_sample_tokenwise_kl, all_settings_in_aug, setting_version_accumilator)

    _fill_aggregated_combinations(result, aug_aggregated_per_sample_tokenwise_kl, setting_version_accumilator,
                                   aug_version_accumilator, ratio, metric_cfg.suffix)

    return result, meta


def _renyi_divergence_maxk_engine(vision_embeddings, target_parts, metric_cfg, cfg,
                                   concat_with_img=True, ripple=False, eps=1e-12):
    result = dict()
    meta = dict()

    alpha = metric_cfg.alpha
    ratio = metric_cfg.ratio
    _, aug_desc_dict = get_augmentations(cfg)

    orig_vision_probs = [_vision_token_probs(e, None, eps) for e in vision_embeddings["orig"]]

    if concat_with_img:
        orig_img_probs = [_img_token_probs(p, None, eps) for p in _orig_img_probs_list(target_parts)]

    setting_version_accumilator = metric_cfg.augmentation_accumilator
    aug_version_accumilator = metric_cfg.augmentation_setting_version_accumilator

    aug_aggregated_per_sample_tokenwise_kl = list()
    for aug, settings in vision_embeddings.items():
        if aug == "orig":
            continue

        all_settings_in_aug = dict()
        for setting_idx, setting in enumerate(settings):
            key = aug_desc_dict[aug][setting_idx]
            all_samples_in_setting_values = list()
            for sample_idx, aug_emb in enumerate(setting):
                aug_probs = _vision_token_probs(aug_emb, None, eps)
                kl = renyi_div_per_token(orig_vision_probs[sample_idx], aug_probs, alpha, eps)

                if concat_with_img:
                    aug_img_probs = _img_token_probs(_aug_img_probs_list(target_parts, aug, setting_idx)[sample_idx], None, eps)
                    img_kl = renyi_div_per_token(orig_img_probs[sample_idx], aug_img_probs, alpha, eps)
                    kl = np.concatenate([kl, img_kl])

                all_samples_in_setting_values.append(kl)

            all_settings_in_aug[str(key)] = all_samples_in_setting_values

        meta[aug] = all_settings_in_aug

        if aug not in result:
            result[aug] = dict()

        if 'none' in setting_version_accumilator:
            _fill_none_accumulated(result[aug], all_settings_in_aug, ratio, ripple, cfg)

        _accumulate_across_settings(aug_aggregated_per_sample_tokenwise_kl, all_settings_in_aug, setting_version_accumilator)

    _fill_aggregated_combinations(result, aug_aggregated_per_sample_tokenwise_kl, setting_version_accumilator,
                                   aug_version_accumilator, ratio, metric_cfg.suffix)

    return result, meta


def _fill_none_accumulated(result_for_aug, all_settings_in_aug, ratio, ripple, cfg):
    if ripple:
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
                        sample_scores.append((-1 * np.mean(np.sort(sliced)[-k_length:])).item())
                    result_for_aug[str(key)] = sample_scores
    else:
        for _ratio in ratio:
            for setting_name, setting_values in all_settings_in_aug.items():
                key = ast.literal_eval(setting_name)
                key['k_ratio'] = _ratio

                sample_scores = list()
                for sample in setting_values:
                    k_length = max(1, int(_ratio * len(sample)))
                    sample_scores.append((-1 * np.mean(np.sort(sample)[-k_length:])).item())
                result_for_aug[str(key)] = sample_scores


def _accumulate_across_settings(aug_aggregated_per_sample_tokenwise_kl, all_settings_in_aug, setting_version_accumilator):
    setting_aggregated_per_sample_tokenwise_kl = dict()
    settings_array = np.array(list(all_settings_in_aug.values()))
    if 'max' in setting_version_accumilator:
        setting_aggregated_per_sample_tokenwise_kl.setdefault('max', list())
        setting_aggregated_per_sample_tokenwise_kl['max'].append(np.max(settings_array, axis=0))
    if 'avg' in setting_version_accumilator:
        setting_aggregated_per_sample_tokenwise_kl.setdefault('avg', list())
        setting_aggregated_per_sample_tokenwise_kl['avg'].append(np.mean(settings_array, axis=0))
    aug_aggregated_per_sample_tokenwise_kl.append(setting_aggregated_per_sample_tokenwise_kl)


def _fill_aggregated_combinations(result, aug_aggregated_per_sample_tokenwise_kl, setting_version_accumilator,
                                   aug_version_accumilator, ratio, suffix):
    final_combinations = dict()
    if 'max' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            if setting_accumilator == 'none':
                continue
            key = f'aggregated_maxed_aug_{setting_accumilator}ed_settings'
            stacked = [v[setting_accumilator] for v in aug_aggregated_per_sample_tokenwise_kl]
            final_combinations[key] = np.max(np.array(stacked), axis=0).tolist()

    if 'avg' in aug_version_accumilator:
        for setting_accumilator in setting_version_accumilator:
            if setting_accumilator == 'none':
                continue
            key = f'aggregated_avged_aug_{setting_accumilator}ed_settings'
            stacked = [v[setting_accumilator] for v in aug_aggregated_per_sample_tokenwise_kl]
            final_combinations[key] = np.mean(np.array(stacked), axis=0).tolist()

    if len(final_combinations) != 0:
        result['aggregated'] = dict()

    for _ratio in ratio:
        for combination, samples in final_combinations.items():
            key = f"Max_{_ratio}_{suffix}_kld_{combination}"
            sample_scores = list()
            for sample in samples:
                k_length = max(1, int(_ratio * len(sample)))
                sample_scores.append(float(-1 * np.mean(np.sort(sample)[-k_length:])))
            result['aggregated'][key] = sample_scores


# =========================================================================
# Distributional baseline metrics (mirrors baseline_metrics.max_prob_gap /
# max_entropy / min_entropy). These are computed on "orig" only - no
# augmentation comparison, just a per-token scalar reduced via top/bottom-k.
# =========================================================================

def vision_max_prob_gap(vision_embeddings, target_parts, concat_with_img=True, eps=1e-12, **_):
    result = list()
    for sample_idx, emb in enumerate(vision_embeddings["orig"]):
        probs = _vision_token_probs(emb, None, eps)
        gap = _gap_per_token(probs)
        if concat_with_img:
            img_probs = _img_token_probs(_orig_img_probs_list(target_parts)[sample_idx], None, eps)
            gap = np.concatenate([gap, _gap_per_token(img_probs)])
        result.append(np.mean(gap).item())
    return result


def vision_max_entropy(vision_embeddings, target_parts, metric_cfg, alpha, concat_with_img=True, eps=1e-12):
    ratio = metric_cfg.ratio
    result = dict()

    entropies = list()
    for sample_idx, emb in enumerate(vision_embeddings["orig"]):
        probs = _vision_token_probs(emb, None, eps)
        entro = _entropy_per_token(probs, alpha)
        if concat_with_img:
            img_probs = _img_token_probs(_orig_img_probs_list(target_parts)[sample_idx], None, eps)
            entro = np.concatenate([entro, _entropy_per_token(img_probs, alpha)])
        entropies.append(entro)

    for _ratio in ratio:
        _key = f"Max_{_ratio*100}% " + metric_cfg.suffix
        result[_key] = list()
        for _entro in entropies:
            k_length = max(1, int(len(_entro) * _ratio))
            topk = np.sort(_entro)[-k_length:]
            result[_key].append(float(-1 * np.mean(topk)))
    return result


def vision_min_entropy(vision_embeddings, target_parts, metric_cfg, alpha, concat_with_img=True, eps=1e-12):
    ratio = metric_cfg.ratio
    result = dict()

    entropies = list()
    for sample_idx, emb in enumerate(vision_embeddings["orig"]):
        probs = _vision_token_probs(emb, None, eps)
        entro = _entropy_per_token(probs, alpha)
        if concat_with_img:
            img_probs = _img_token_probs(_orig_img_probs_list(target_parts)[sample_idx], None, eps)
            entro = np.concatenate([entro, _entropy_per_token(img_probs, alpha)])
        entropies.append(entro)

    for _ratio in ratio:
        _key = str({'k_ratio': _ratio})
        result[_key] = list()
        for _entro in entropies:
            k_length = max(1, int(len(_entro) * _ratio))
            bottomk = np.sort(_entro)[:k_length]
            result[_key].append(float(-1 * np.mean(bottomk)))
    return result


# =========================================================================
# Dispatcher
# =========================================================================

# (opt_in_name, cfg_block_name, alpha, ripple)
_KL_DIV_SPECS = [
    ("vision_max_k_no_norn_kl_div", "max_k_no_norn_kl_div", None, False),
    ("vision_max_k_renyi_05_kl_div", "max_k_renyi_05_kl_div", 0.5, False),
    ("vision_max_k_renyi_1_kl_div", "max_k_renyi_1_kl_div", 1, False),
    ("vision_max_k_renyi_2_kl_div", "max_k_renyi_2_kl_div", 2, False),
    ("vision_max_k_renyi_inf_kl_div", "max_k_renyi_inf_kl_div", "inf", False),
    ("vision_max_k_no_norn_kl_div_ripple", "max_k_no_norn_kl_div_ripple", None, True),
    ("vision_max_k_renyi_05_kl_div_ripple", "max_k_renyi_05_kl_div_ripple", 0.5, True),
    ("vision_max_k_renyi_1_kl_div_ripple", "max_k_renyi_1_kl_div_ripple", 1, True),
    ("vision_max_k_renyi_2_kl_div_ripple", "max_k_renyi_2_kl_div_ripple", 2, True),
    ("vision_max_k_renyi_inf_kl_div_ripple", "max_k_renyi_inf_kl_div_ripple", "inf", True),
]

# (opt_in_name, cfg_block_name, ripple) -- alpha is read from the cfg block itself
_RENYI_DIV_SPECS = [
    ("vision_max_k_renyi_divergence_025", "max_k_renyi_divergence_025", False),
    ("vision_max_k_renyi_divergence_05", "max_k_renyi_divergence_05", False),
    ("vision_max_k_renyi_divergence_2", "max_k_renyi_divergence_2", False),
    ("vision_max_k_renyi_divergence_4", "max_k_renyi_divergence_4", False),
    ("vision_max_k_renyi_divergence_025_ripple", "max_k_renyi_divergence_025_ripple", True),
    ("vision_max_k_renyi_divergence_05_ripple", "max_k_renyi_divergence_05_ripple", True),
    ("vision_max_k_renyi_divergence_2_ripple", "max_k_renyi_divergence_2_ripple", True),
    ("vision_max_k_renyi_divergence_4_ripple", "max_k_renyi_divergence_4_ripple", True),
]

# (opt_in_name, cfg_block_name, alpha, fn)
_ENTROPY_SPECS = [
    ("vision_max_k_renyi_1_entro", "max_k_renyi_1_entro", 1, vision_max_entropy),
    ("vision_max_k_renyi_2_entro", "max_k_renyi_2_entro", 2, vision_max_entropy),
    ("vision_max_k_renyi_05_entro", "max_k_renyi_05_entro", 0.5, vision_max_entropy),
    ("vision_min_k_renyi_1_entro", "min_k_renyi_1_entro", 1, vision_min_entropy),
    ("vision_min_k_renyi_2_entro", "min_k_renyi_2_entro", 2, vision_min_entropy),
    ("vision_min_k_renyi_05_entro", "min_k_renyi_05_entro", 0.5, vision_min_entropy),
]

_VARIANTS = (("vision", False), ("vision_img_concat", True))


def get_vision_embedding_metrics(vision_embeddings, target_parts, cfg):
    """
    vision_embeddings: batch_vision_embeddings dict returned by mod_infer_batch_hulu
        {"orig": [tensor, ...], aug_name: [[tensor, ...] per setting]}
    target_parts: target_parts dict returned by mod_infer_batch_hulu (only used for
        the "vision_img_concat" variant; ignored if "img" is not in cfg.img_metrics.parts)
    cfg: full hydra config. Gated by cfg.img_metrics.metrics_to_use containing a
        "vision_"-prefixed metric name; reuses the SAME per-metric config blocks as
        the text-token metrics (e.g. cfg.img_metrics.max_k_no_norn_kl_div).

    Returns (pred, meta) with up to two top-level keys ("vision", "vision_img_concat"),
    each shaped exactly like a normal part's pred dict (kld_metrics / renyi_div_metrics /
    baseline_metrics), so callers can merge them straight into `preds` alongside the
    existing "img" / "inst_desc" / "img_inst_desc" parts with no change to eval.py.
    """
    metrics_to_use = cfg.img_metrics.metrics_to_use
    has_img_part = "img" in cfg.img_metrics.parts

    pred = {"vision": dict(), "vision_img_concat": dict()}
    meta = {"vision": dict(), "vision_img_concat": dict()}

    def variants():
        for variant, concat in _VARIANTS:
            if concat and not has_img_part:
                continue
            yield variant, concat

    for opt_name, cfg_name, alpha, ripple in _KL_DIV_SPECS:
        if opt_name not in metrics_to_use:
            continue
        metric_cfg = getattr(cfg.img_metrics, cfg_name)
        for variant, concat in variants():
            pred[variant].setdefault('kld_metrics', dict())
            meta[variant].setdefault('kld_metrics', dict())
            _pred, _meta = _kl_div_maxk_engine(vision_embeddings, target_parts, metric_cfg, cfg,
                                                alpha=alpha, concat_with_img=concat, ripple=ripple)
            pred[variant]['kld_metrics'][opt_name] = _pred
            meta[variant]['kld_metrics'][f"{opt_name}_tkn_vals"] = _meta

    for opt_name, cfg_name, ripple in _RENYI_DIV_SPECS:
        if opt_name not in metrics_to_use:
            continue
        metric_cfg = getattr(cfg.img_metrics, cfg_name)
        for variant, concat in variants():
            pred[variant].setdefault('renyi_div_metrics', dict())
            meta[variant].setdefault('renyi_div_metrics', dict())
            _pred, _meta = _renyi_divergence_maxk_engine(vision_embeddings, target_parts, metric_cfg, cfg,
                                                          concat_with_img=concat, ripple=ripple)
            pred[variant]['renyi_div_metrics'][opt_name] = _pred
            meta[variant]['renyi_div_metrics'][f"{opt_name}_tkn_vals"] = _meta

    if "vision_max_prob_gap" in metrics_to_use:
        for variant, concat in variants():
            pred[variant].setdefault('baseline_metrics', dict())
            pred[variant]['baseline_metrics']["vision_max_prob_gap"] = vision_max_prob_gap(
                vision_embeddings, target_parts, concat_with_img=concat)

    for opt_name, cfg_name, alpha, fn in _ENTROPY_SPECS:
        if opt_name not in metrics_to_use:
            continue
        metric_cfg = getattr(cfg.img_metrics, cfg_name)
        for variant, concat in variants():
            pred[variant].setdefault('baseline_metrics', dict())
            pred[variant]['baseline_metrics'][opt_name] = fn(
                vision_embeddings, target_parts, metric_cfg, alpha, concat_with_img=concat)

    pred = {k: v for k, v in pred.items() if len(v) > 0}
    meta = {k: v for k, v in meta.items() if len(v) > 0}

    return pred, meta
