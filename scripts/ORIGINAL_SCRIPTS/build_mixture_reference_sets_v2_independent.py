"""
Corrected version of build_mixture_reference_sets.py -- see
../../../additional_ICLR_experiments/KNOWN_ISSUE_exact_nonmember_reference.md (in
VLM_MIA_STUDY_Archive_Data) for the full writeup of why the original construction (p * target's
own exact non-members + (1-p) * ShareGPT) breaks the tuning signal's sign for p < ~0.9: since the
"target-like" component is literally the same images already in the target set's non-member
half, the gap's non-member coefficient (0.5-p) crosses exactly zero at p=0.5 and flips sign
beyond it, rather than genuinely damping.

Fix: for p=0..90%, the "target-like" portion of the reference is now drawn from a genuinely
*independent* pool of flickr non-members -- same distribution as the target's own non-members,
but different specific images -- matching how the original Figure 3 ("Estimated" =
sharegpt0.5_flickr0.5_KN) was actually built (verified: only 23% of its reference overlapped the
target's own non-members, not the ~50% you'd expect from exact-duplication at p=0.5).

Independent pool source: FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json (300
candidates) = the target's own 150 non-members + 150 genuinely disjoint ones (verified via image
content hashing, not path strings -- the disjoint 150 were confirmed never used anywhere in the
target set). Only those 150 disjoint images are used here.

p=100% is intentionally left unchanged from the original build -- that endpoint is *meant* to be
the exact target non-members (matching exact_flickr_KN / "True optimal"), was never affected by
the bug (coefficient = -0.5 there regardless), and its data is already fully computed -- so this
script does not touch it; the sweep script reuses the original reference_dataset_p100.parquet
directly instead of a v2 copy.

Usage:
    conda activate med_vlm_mia_venv
    python scripts/ORIGINAL_SCRIPTS/build_mixture_reference_sets_v2_independent.py
"""
import hashlib
import os
import json

import numpy as np
from datasets import Dataset, concatenate_datasets
from datasets.features import Image as HFImage
from PIL import Image as PILImage

SEED = 0
REFERENCE_SET_SIZE = 150

ICLR_EXPERIMENTS_DIR = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/additional_ICLR_experiments"
NON_MEMBER_TARGET_PARQUET = os.path.join(ICLR_EXPERIMENTS_DIR, "datasets", "non_member_target_dataset.parquet")
SHAREGPT_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/share_gpt/global_nonmember_fullset.json"
FLICKR_NONMEMBER_POOL_JSON = "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_DATASETS/flickr/flickr_nonmember_subset.json"

OUT_DIR = os.path.join(ICLR_EXPERIMENTS_DIR, "datasets", "mixture_reference_sets_v2_independent")

# p=100% deliberately excluded -- unchanged/reused from the original build, see module docstring.
MIXTURE_FRACTIONS = np.linspace(0.0, 0.9, 10)


def force_reference_labels(ds):
    for col in ("tune_label", "label"):
        if col in ds.column_names:
            ds = ds.remove_columns(col)
    ds = ds.add_column("tune_label", [0] * len(ds))
    ds = ds.add_column("label", [0] * len(ds))
    return ds


def img_hash(pil_img):
    return hashlib.md5(pil_img.convert("RGB").tobytes()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    target_nonmembers = Dataset.from_parquet(NON_MEMBER_TARGET_PARQUET)
    target_hashes = {img_hash(target_nonmembers[i]["image"]) for i in range(len(target_nonmembers))}
    print(f"Target's own non-member set: {len(target_hashes)} unique images")

    pool = json.load(open(FLICKR_NONMEMBER_POOL_JSON))
    disjoint_entries = [x for x in pool if img_hash(PILImage.open(x["image"])) not in target_hashes]
    print(f"Independent (disjoint-from-target) flickr non-member pool: {len(disjoint_entries)} / {len(pool)}")
    assert len(disjoint_entries) >= REFERENCE_SET_SIZE, (
        f"Need {REFERENCE_SET_SIZE} disjoint flickr non-members, only found {len(disjoint_entries)}"
    )

    disjoint_flickr = Dataset.from_list(disjoint_entries).cast_column("image", HFImage(decode=True))
    disjoint_flickr = disjoint_flickr.shuffle(seed=SEED)  # canonical fixed order

    with open(SHAREGPT_POOL_JSON, "r") as f:
        sharegpt_pool = json.load(f)
    rng = np.random.default_rng(SEED)
    sharegpt_idxs = rng.choice(len(sharegpt_pool), size=REFERENCE_SET_SIZE, replace=False)
    sharegpt_subset = Dataset.from_list([sharegpt_pool[i] for i in sharegpt_idxs])
    sharegpt_subset = sharegpt_subset.cast_column("image", HFImage(decode=True))

    print(f"\n{'p':>6}  {'n_flickr':>8}  {'n_sharegpt':>10}  out_path")
    for p in MIXTURE_FRACTIONS:
        n_flickr = int(round(p * REFERENCE_SET_SIZE))
        n_sharegpt = REFERENCE_SET_SIZE - n_flickr

        parts = []
        if n_flickr > 0:
            parts.append(disjoint_flickr.select(range(n_flickr)).remove_columns(
                [c for c in disjoint_flickr.column_names if c not in ("image", "label", "tune_label")]
            ))
        if n_sharegpt > 0:
            parts.append(sharegpt_subset.select(range(n_sharegpt)))

        mixed = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        mixed = force_reference_labels(mixed)

        pp = int(round(p * 100))
        out_path = os.path.join(OUT_DIR, f"reference_dataset_p{pp:03d}.parquet")
        mixed.to_parquet(out_path)
        print(f"{p:6.2f}  {n_flickr:8d}  {n_sharegpt:10d}  {out_path}")

    print(f"\nDone -- {len(MIXTURE_FRACTIONS)} mixture reference sets (p=0..90%) written to {OUT_DIR}")
    print("p=100% intentionally not rebuilt -- reuse the original reference_dataset_p100.parquet directly.")


if __name__ == "__main__":
    main()
