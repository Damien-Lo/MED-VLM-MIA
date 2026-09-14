"""
Third attempt at the mixture reference sets -- see
/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/README.txt for the full
history (v1 exact-duplication, v2 independent-but-excluded pool, both failed to reproduce the
dampening effect; figure3_reproduction/ confirmed the real methodology on real data).

Fix vs v2: v2 explicitly EXCLUDED the target set's own non-members from the candidate pool
(sampling only the 150 "genuinely disjoint" images), guaranteeing 0% overlap by construction.
Verified directly against the real sharegpt0.5_flickr0.5_KN reference set that this is NOT what
the original methodology did -- it sampled *unrestricted* from the full 300-candidate pool
(target's own 150 + 150 disjoint), landing at 150/300 = 50% of the reference matching that pool,
with 68/150 (~45%) of the flickr-half incidentally overlapping the target's own non-members by
chance, not by exclusion. This script removes that exclusion.

Reference set SIZE: also discovered the real p=0.5 experiment used a 300-row reference (150
ShareGPT + 150 flickr, i.e. a full 150-strong block per source rather than a proportional split
of a fixed 150 total), while the real p=1.0 experiment (exact_flickr_KN) used a 150-row reference
(matching its single source exactly). There's no unambiguous way to generalize the 300-row
"one full block per active source" convention to intermediate p -- so:
  - p=0.5 reuses the REAL reference_dataset.parquet verbatim (300 rows, byte-identical to what
    produced the confirmed-correct Figure 3 "Estimated" curve) -- maximum fidelity for this point.
  - p=0.0 and p=1.0 reuse the existing 150-row pure-ShareGPT / pure-exact-nonmember sets.
  - p=0.1..0.4, 0.6..0.9 use a 150-row reference (matching the single-source endpoints), with
    round(p*150) drawn *unrestricted* from the same 300-candidate flickr pool + the remainder
    ShareGPT -- i.e. the v2 construction, minus the exclusion bug. This is a reasonable, clearly
    documented choice given the real ambiguity, not a silent guess.

Usage:
    conda activate med_vlm_mia_venv
    python scripts/ORIGINAL_SCRIPTS/build_mixture_reference_sets_v3_unrestricted.py
"""
import os
import json
import shutil

import numpy as np
from datasets import Dataset, concatenate_datasets
from datasets.features import Image as HFImage

SEED = 0
REFERENCE_SET_SIZE = 150

ICLR_DIR = "/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results"
DATASETS_DIR = os.path.join(ICLR_DIR, "datasets")

SHAREGPT_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/share_gpt/global_nonmember_fullset.json"
FLICKR_NONMEMBER_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json"  # 300 candidates, UNRESTRICTED (includes target's own 150)

REAL_P050_REFERENCE_PARQUET = (
    "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/"
    "hyperparam_tuning/flickr_pretrain_member_ratio_0.5__sharegpt0.5_flickr0.5_KN/run_1/gn_set0/"
    "datasets/reference_dataset.parquet"
)  # confirmed: reproduces the published Figure 3 "Estimated" curve exactly (sigma* = 6.27e-03)

OUT_DIR = os.path.join(DATASETS_DIR, "mixture_reference_sets_v3_unrestricted")

MIXTURE_FRACTIONS_TO_BUILD = [10, 20, 30, 40, 60, 70, 80, 90]  # p=0/50/100 handled by direct reuse below


def force_reference_labels(ds):
    for col in ("tune_label", "label"):
        if col in ds.column_names:
            ds = ds.remove_columns(col)
    ds = ds.add_column("tune_label", [0] * len(ds))
    ds = ds.add_column("label", [0] * len(ds))
    return ds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- p=0.5: reuse the REAL reference set verbatim ---
    real_p050 = Dataset.from_parquet(REAL_P050_REFERENCE_PARQUET)
    real_p050 = force_reference_labels(real_p050)
    real_p050.to_parquet(os.path.join(OUT_DIR, "reference_dataset_p050.parquet"))
    print(f"p=0.50: reused REAL reference_dataset.parquet verbatim -- {len(real_p050)} rows")

    # --- p=0.0: reuse existing pure-ShareGPT set (150 rows) ---
    p000_src = os.path.join(DATASETS_DIR, "mixture_reference_sets", "reference_dataset_p000.parquet")
    shutil.copy(p000_src, os.path.join(OUT_DIR, "reference_dataset_p000.parquet"))
    print(f"p=0.00: reused existing pure-ShareGPT reference set unchanged")

    # --- p=1.0: reuse existing exact-target-nonmember set (150 rows) ---
    p100_src = os.path.join(DATASETS_DIR, "mixture_reference_sets", "reference_dataset_p100.parquet")
    shutil.copy(p100_src, os.path.join(OUT_DIR, "reference_dataset_p100.parquet"))
    print(f"p=1.00: reused existing exact-target-nonmember reference set unchanged")

    # --- p=0.1..0.4, 0.6..0.9: 150-row reference, unrestricted flickr pool sampling ---
    pool = json.load(open(FLICKR_NONMEMBER_POOL_JSON))
    flickr_pool = Dataset.from_list(pool).cast_column("image", HFImage(decode=True))
    flickr_pool = flickr_pool.shuffle(seed=SEED)  # canonical fixed order, UNRESTRICTED (no exclusion)

    with open(SHAREGPT_POOL_JSON, "r") as f:
        sharegpt_pool = json.load(f)
    rng = np.random.default_rng(SEED)
    sharegpt_idxs = rng.choice(len(sharegpt_pool), size=REFERENCE_SET_SIZE, replace=False)
    sharegpt_subset = Dataset.from_list([sharegpt_pool[i] for i in sharegpt_idxs])
    sharegpt_subset = sharegpt_subset.cast_column("image", HFImage(decode=True))

    print(f"\n{'p':>6}  {'n_flickr':>8}  {'n_sharegpt':>10}  out_path")
    for pp in MIXTURE_FRACTIONS_TO_BUILD:
        p = pp / 100.0
        n_flickr = int(round(p * REFERENCE_SET_SIZE))
        n_sharegpt = REFERENCE_SET_SIZE - n_flickr

        parts = []
        if n_flickr > 0:
            parts.append(flickr_pool.select(range(n_flickr)).remove_columns(
                [c for c in flickr_pool.column_names if c not in ("image", "label", "tune_label")]
            ))
        if n_sharegpt > 0:
            parts.append(sharegpt_subset.select(range(n_sharegpt)))

        mixed = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        mixed = force_reference_labels(mixed)

        out_path = os.path.join(OUT_DIR, f"reference_dataset_p{pp:03d}.parquet")
        mixed.to_parquet(out_path)
        print(f"{p:6.2f}  {n_flickr:8d}  {n_sharegpt:10d}  {out_path}")

    print(f"\nDone -- 11 mixture reference sets (p=0,10,...,100) available in {OUT_DIR}")
    print("(p=0/50/100 are direct file reuses, not freshly sampled -- see script docstring)")


if __name__ == "__main__":
    main()
