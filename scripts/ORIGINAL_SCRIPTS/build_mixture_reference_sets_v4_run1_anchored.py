"""
Fourth attempt at the mixture reference sets -- see
/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/README.txt for the full
history (v1 exact-duplication, v2 independent-but-excluded pool, v3 unrestricted-pool sampling --
all confirmed the divergence *computation* is correct, but none anchored EVERY sweep point to
run_1's own validated data, so p=0.5 in v3 still didn't reproduce Figure 3 exactly (it reused
run_1's real reference_dataset.parquet, but paired it with an independently-resampled target set,
only ~51% overlapping run_1's own target set).

Fix vs v3: anchor BOTH the target set and the p=0.5 reference set to run_1's own frozen files
byte-for-byte, so p=0.5 is guaranteed to reproduce the published Figure 3 exactly (both "Estimated"
and "True optimal" curves). Every other p is then built by *modifying* that same anchor rather
than resampling independently:
  - flickr half: for p <= 0.5 (n_flickr <= 150), take a deterministic subset of run_1's own 150
    flickr reference rows (first n_flickr in file order). For p > 0.5 (n_flickr > 150), keep all
    150 of run_1's flickr rows as a fixed core and supplement the remainder from the broader
    unrestricted flickr_nonmember_subset.json pool (excluding those same 150, so no duplicates).
  - sharegpt half: same idea, subset/supplement run_1's own 150 sharegpt reference rows against
    the broader ShareGPT global_nonmember_fullset.json pool.
  - reference set size is always 300 (matching run_1's real reference set), not 150 like v3's
    intermediate points -- so every p point in this sweep shares the same total size and, where
    possible, the same literal rows as its neighbors, differing only in the flickr/sharegpt split.

This means every p from 0..100 nests around the p=0.5 anchor: as p moves away from 0.5 in either
direction, rows are swapped out/in incrementally rather than the whole set being redrawn, which
should make the sweep's shape far more internally consistent and directly traceable back to the
one confirmed-correct data point.

Usage:
    conda activate med_vlm_mia_venv
    python scripts/ORIGINAL_SCRIPTS/build_mixture_reference_sets_v4_run1_anchored.py
"""
import os
import json
import shutil

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from datasets.features import Image as HFImage

SEED = 0
REFERENCE_SET_SIZE = 300  # matches run_1's real reference set (150 flickr + 150 sharegpt)
CORE_SIZE = 150  # run_1's own per-source row count

ICLR_DIR = "/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results"
DATASETS_DIR = os.path.join(ICLR_DIR, "datasets")

SHAREGPT_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/share_gpt/global_nonmember_fullset.json"
FLICKR_NONMEMBER_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json"

RUN1_DIR = (
    "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/"
    "hyperparam_tuning/flickr_pretrain_member_ratio_0.5__sharegpt0.5_flickr0.5_KN/run_1/gn_set0/datasets"
)
RUN1_TARGET_PARQUET = os.path.join(RUN1_DIR, "target_dataset.parquet")
RUN1_REFERENCE_PARQUET = os.path.join(RUN1_DIR, "reference_dataset.parquet")

OUT_DIR = os.path.join(DATASETS_DIR, "mixture_reference_sets_v4_run1_anchored")
OUT_TARGET_PARQUET = os.path.join(DATASETS_DIR, "target_dataset_run1_exact.parquet")

MIXTURE_FRACTIONS_TO_BUILD = [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]  # p=50 handled by direct reuse


def force_reference_labels(ds):
    for col in ("tune_label", "label"):
        if col in ds.column_names:
            ds = ds.remove_columns(col)
    ds = ds.add_column("tune_label", [0] * len(ds))
    ds = ds.add_column("label", [0] * len(ds))
    return ds


def raw_image_paths(parquet_path):
    """Read a dataset's image paths via pandas (raw struct, never auto-decoded to PIL) --
    the `datasets` library auto-decodes the Image feature to a PIL object on row access, which
    isn't subscriptable, so path extraction has to go through pandas instead."""
    return pd.read_parquet(parquet_path)["image"].apply(lambda x: x["path"]).tolist()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Target set: reuse run_1's real target set verbatim (shared by ALL p values) ---
    run1_target = Dataset.from_parquet(RUN1_TARGET_PARQUET)
    run1_target.to_parquet(OUT_TARGET_PARQUET)
    print(f"Target set: reused run_1's real target_dataset.parquet verbatim -- {len(run1_target)} rows -> {OUT_TARGET_PARQUET}")

    # --- p=0.5: reuse run_1's real reference set verbatim ---
    run1_ref = Dataset.from_parquet(RUN1_REFERENCE_PARQUET)
    run1_ref = force_reference_labels(run1_ref)
    run1_ref.to_parquet(os.path.join(OUT_DIR, "reference_dataset_p050.parquet"))
    print(f"p=0.50: reused run_1's real reference_dataset.parquet verbatim -- {len(run1_ref)} rows")

    # Split run_1's reference set into its flickr-core and sharegpt-core (150 each)
    run1_ref_paths = raw_image_paths(RUN1_REFERENCE_PARQUET)
    is_flickr = ["flickr" in p.lower() for p in run1_ref_paths]
    flickr_core = run1_ref.select([i for i, f in enumerate(is_flickr) if f])
    sharegpt_core = run1_ref.select([i for i, f in enumerate(is_flickr) if not f])
    assert len(flickr_core) == CORE_SIZE and len(sharegpt_core) == CORE_SIZE, (
        f"expected {CORE_SIZE}/{CORE_SIZE} flickr/sharegpt split, got {len(flickr_core)}/{len(sharegpt_core)}"
    )
    flickr_core_paths = set(p for p, f in zip(run1_ref_paths, is_flickr) if f)
    sharegpt_core_paths = set(p for p, f in zip(run1_ref_paths, is_flickr) if not f)
    print(f"run_1 reference core: {len(flickr_core)} flickr + {len(sharegpt_core)} sharegpt")

    # --- Supplement pools for p far enough from 0.5 that the 150-row core isn't enough ---
    # Pool JSONs store "image" as a plain path string (not the {"path":..,"bytes":..} struct the
    # parquet-loaded datasets use) -- cast_column("image", HFImage(decode=True)) accepts either.
    flickr_pool_all = json.load(open(FLICKR_NONMEMBER_POOL_JSON))
    flickr_pool_extra = [r for r in flickr_pool_all if r["image"] not in flickr_core_paths]
    flickr_pool_extra_ds = Dataset.from_list(flickr_pool_extra).cast_column("image", HFImage(decode=True))
    flickr_pool_extra_ds = flickr_pool_extra_ds.shuffle(seed=SEED)  # fixed deterministic order
    flickr_pool_extra_ds = force_reference_labels(flickr_pool_extra_ds)  # match core's exact schema up front

    sharegpt_pool_all = json.load(open(SHAREGPT_POOL_JSON))
    sharegpt_pool_extra = [r for r in sharegpt_pool_all if r["image"] not in sharegpt_core_paths]
    rng = np.random.default_rng(SEED)
    sharegpt_extra_idxs = rng.permutation(len(sharegpt_pool_extra))
    sharegpt_pool_extra_ds = Dataset.from_list([sharegpt_pool_extra[i] for i in sharegpt_extra_idxs]).cast_column(
        "image", HFImage(decode=True)
    )
    sharegpt_pool_extra_ds = force_reference_labels(sharegpt_pool_extra_ds)  # match core's exact schema up front

    print(f"Supplement pools available: {len(flickr_pool_extra_ds)} extra flickr, {len(sharegpt_pool_extra_ds)} extra sharegpt")

    keep_cols = ["image", "label", "tune_label"]

    def trim_cols(ds):
        return ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    print(f"\n{'p':>6}  {'n_flickr':>8}  {'n_sharegpt':>10}  {'flickr_src':>22}  {'sharegpt_src':>22}  out_path")
    for pp in MIXTURE_FRACTIONS_TO_BUILD:
        p = pp / 100.0
        n_flickr = int(round(p * REFERENCE_SET_SIZE))
        n_sharegpt = REFERENCE_SET_SIZE - n_flickr

        # --- flickr half: subset of the core (p<=0.5), or core + supplement (p>0.5) ---
        if n_flickr <= CORE_SIZE:
            flickr_part = trim_cols(flickr_core.select(range(n_flickr))) if n_flickr > 0 else None
            flickr_src = f"core[:{n_flickr}]"
        else:
            n_supp = n_flickr - CORE_SIZE
            flickr_part = concatenate_datasets(
                [trim_cols(flickr_core), trim_cols(flickr_pool_extra_ds.select(range(n_supp)))]
            )
            flickr_src = f"core+{n_supp}supp"

        # --- sharegpt half: same pattern ---
        if n_sharegpt <= CORE_SIZE:
            sharegpt_part = trim_cols(sharegpt_core.select(range(n_sharegpt))) if n_sharegpt > 0 else None
            sharegpt_src = f"core[:{n_sharegpt}]"
        else:
            n_supp = n_sharegpt - CORE_SIZE
            sharegpt_part = concatenate_datasets(
                [trim_cols(sharegpt_core), trim_cols(sharegpt_pool_extra_ds.select(range(n_supp)))]
            )
            sharegpt_src = f"core+{n_supp}supp"

        parts = [x for x in (flickr_part, sharegpt_part) if x is not None]
        mixed = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        mixed = force_reference_labels(mixed)
        assert len(mixed) == REFERENCE_SET_SIZE, f"p={pp}: expected {REFERENCE_SET_SIZE} rows, got {len(mixed)}"

        out_path = os.path.join(OUT_DIR, f"reference_dataset_p{pp:03d}.parquet")
        mixed.to_parquet(out_path)
        print(f"{p:6.2f}  {n_flickr:8d}  {n_sharegpt:10d}  {flickr_src:>22}  {sharegpt_src:>22}  {out_path}")

    print(f"\nDone -- 11 mixture reference sets (p=0,10,...,100) available in {OUT_DIR}")
    print("(p=0.5 is a byte-identical copy of run_1's real reference_dataset.parquet; all others")
    print(" nest around it by subsetting/supplementing its own 150-flickr/150-sharegpt core)")


if __name__ == "__main__":
    main()
