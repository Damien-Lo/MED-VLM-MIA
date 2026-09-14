"""
Builds a family of reference sets that interpolate between "pure ShareGPT" and "pure flickr
target non-members" -- one reference_dataset_p{NN}.parquet per mixture fraction p in
MIXTURE_FRACTIONS, for the distributional-difference sweep discussed for the LLaVA/flickr
hyperparam_tuning rerun.

Reuses the exact frozen sets from the original run:
  - non_member_target_dataset.parquet: the target set's own 150 flickr non-members (used
    directly at higher p, not a separate disjoint flickr-non-member pool -- confirmed this is
    what's wanted, since the goal is to see what happens as the reference set becomes
    literally the target non-member set, not just "same distribution, different samples").
  - the ShareGPT nonmember pool json (57421 samples) as the other end of the mixture.

For a given p, the reference set is 150 rows: round(p*150) from the flickr target non-members
+ the remainder from ShareGPT. Both sides are sampled from a single canonical shuffled order
fixed once up front (SEED below), then sliced by count -- so the family is *nested*: each
higher-p mixture's flickr portion is a superset of every lower-p mixture's, and each lower-p
mixture's ShareGPT portion is a superset of every higher-p mixture's. That isolates the
mixture-fraction variable itself as the only thing changing between adjacent curves, rather
than also introducing fresh sampling noise at every point.

Every row gets label=0 / tune_label=0 forced (both pools are known non-members by
construction), matching build_and_save_reference_set's convention in the main pipeline.

Usage:
    conda activate med_vlm_mia_venv
    python scripts/ORIGINAL_SCRIPTS/build_mixture_reference_sets.py
"""
import os
import json
import numpy as np
from datasets import Dataset, concatenate_datasets
from datasets.features import Image as HFImage

SEED = 0
REFERENCE_SET_SIZE = 150  # capped at the flickr non-member pool size (150) so p=1.0 needs no duplication

ICLR_EXPERIMENTS_DIR = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/additional_ICLR_experiments"

# Copied over from the original .../FINAL_RESULTS/llava/flickr/hyperparam_tuning/
# flickr_pretrain_member_ratio_0.5__sharegpt_KN/run_one/gn_set0/datasets/ run (originals left
# untouched there) -- see ICLR_EXPERIMENTS_DIR/datasets/ for target_dataset.parquet,
# member_target_dataset.parquet, and original_sharegpt_reference_dataset.parquet too.
NON_MEMBER_TARGET_PARQUET = os.path.join(ICLR_EXPERIMENTS_DIR, "datasets", "non_member_target_dataset.parquet")
SHAREGPT_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/share_gpt/global_nonmember_fullset.json"

OUT_DIR = os.path.join(ICLR_EXPERIMENTS_DIR, "datasets", "mixture_reference_sets")

MIXTURE_FRACTIONS = np.linspace(0.0, 1.0, 11)  # 0%, 10%, ..., 100% -- 11 points, both ends included


def force_reference_labels(ds):
    for col in ("tune_label", "label"):
        if col in ds.column_names:
            ds = ds.remove_columns(col)
    ds = ds.add_column("tune_label", [0] * len(ds))
    ds = ds.add_column("label", [0] * len(ds))
    return ds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    flickr_nonmembers = Dataset.from_parquet(NON_MEMBER_TARGET_PARQUET)
    assert len(flickr_nonmembers) == REFERENCE_SET_SIZE, (
        f"Expected {REFERENCE_SET_SIZE} flickr target non-members, got {len(flickr_nonmembers)} -- "
        "adjust REFERENCE_SET_SIZE if the target set size has changed."
    )
    flickr_nonmembers = flickr_nonmembers.shuffle(seed=SEED)  # canonical fixed order

    with open(SHAREGPT_POOL_JSON, "r") as f:
        sharegpt_pool = json.load(f)
    rng = np.random.default_rng(SEED)
    sharegpt_idxs = rng.choice(len(sharegpt_pool), size=REFERENCE_SET_SIZE, replace=False)
    sharegpt_subset = Dataset.from_list([sharegpt_pool[i] for i in sharegpt_idxs])  # canonical fixed order
    sharegpt_subset = sharegpt_subset.cast_column("image", HFImage(decode=True))

    print(f"{'p':>6}  {'n_flickr':>8}  {'n_sharegpt':>10}  out_path")
    for p in MIXTURE_FRACTIONS:
        n_flickr = int(round(p * REFERENCE_SET_SIZE))
        n_sharegpt = REFERENCE_SET_SIZE - n_flickr

        parts = []
        if n_flickr > 0:
            parts.append(flickr_nonmembers.select(range(n_flickr)).remove_columns(
                [c for c in flickr_nonmembers.column_names if c not in ("image", "label", "tune_label")]
            ))
        if n_sharegpt > 0:
            parts.append(sharegpt_subset.select(range(n_sharegpt)))

        mixed = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        mixed = force_reference_labels(mixed)

        pp = int(round(p * 100))
        out_path = os.path.join(OUT_DIR, f"reference_dataset_p{pp:03d}.parquet")
        mixed.to_parquet(out_path)
        print(f"{p:6.2f}  {n_flickr:8d}  {n_sharegpt:10d}  {out_path}")

    print(f"\nDone -- {len(MIXTURE_FRACTIONS)} mixture reference sets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
