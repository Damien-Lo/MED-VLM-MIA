"""
Builds the member/non-member/reference parquet files for the CVPR-redo LLaVA + MiniGPT
fresh-inference run, from the archive's own byte-for-byte-anchored run_1/gn_set0 data --
the one genuinely-valid 50/50 flickr(non-member-distribution)/sharegpt mixture-reference
tuning run (see scripts/ORIGINAL_SCRIPTS/build_mixture_reference_sets_v4_run1_anchored.py's
docstring for how run_1 was validated to exactly reproduce the published Figure 3 at p=0.5).

target_dataset.parquet (300 rows: 150 member/label=1, 150 non-member/label=0) is split into
member_target_dataset.parquet / non_member_target_dataset.parquet; reference_dataset.parquet
(300 rows, 50% flickr-distribution / 50% ShareGPT) is copied as-is. This gives the exact same
{member, non-member, reference} triple the endoscopy/mri modern-pipeline scripts consume via
data.member_dataset / data.nonmember_dataset / data.reference_datasets_list.

Usage: conda activate med_vlm_mia_venv; python build_cvpr_datasets.py

BUG FIX (2026-09-24): the member/non-member split was originally done via pandas
(pd.read_parquet -> filter -> to_parquet), which silently strips the HuggingFace `datasets`
library's Image feature encoding (the 'image' column round-trips as a generic struct instead
of the datasets.Image feature type). This caused mia.py's concatenate_datasets(target,
reference) to fail at tuning time with "features can't be aligned ... has unexpected type -
Image(mode=None, decode=True, id=None)", since reference_dataset.parquet (copied byte-for-
byte via shutil.copy2, never touched by pandas) kept the correct Image feature encoding but
member/non_member_target_dataset.parquet didn't. Fixed by doing the split with the `datasets`
library itself (load_dataset("parquet", ...) + .filter() + .to_parquet()), which preserves
feature types through the split, exactly as shutil.copy2 does for files it doesn't touch.
"""
import shutil
from pathlib import Path

from datasets import load_dataset

RUN1_DIR = Path(
    "/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/flickr/"
    "hyperparam_tuning/flickr_pretrain_member_ratio_0.5__sharegpt0.5_flickr0.5_KN/run_1/gn_set0/datasets"
)
OUT_DIR = Path("/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/llava-flickr/run0/datasets")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target = load_dataset("parquet", data_files=str(RUN1_DIR / "target_dataset.parquet"), split="train")
    assert len(target) == 300, f"expected 300 target rows, got {len(target)}"
    label_counts = {v: target["label"].count(v) for v in set(target["label"])}
    assert label_counts == {1: 150, 0: 150}, f"expected 150/150 member/non-member split, got {label_counts}"

    member = target.filter(lambda ex: ex["label"] == 1)
    nonmember = target.filter(lambda ex: ex["label"] == 0)
    member.to_parquet(str(OUT_DIR / "member_target_dataset.parquet"))
    nonmember.to_parquet(str(OUT_DIR / "non_member_target_dataset.parquet"))
    print(f"wrote {len(member)} member rows -> {OUT_DIR / 'member_target_dataset.parquet'} (features={member.features['image']})")
    print(f"wrote {len(nonmember)} non-member rows -> {OUT_DIR / 'non_member_target_dataset.parquet'} (features={nonmember.features['image']})")

    shutil.copy2(RUN1_DIR / "reference_dataset.parquet", OUT_DIR / "reference_dataset.parquet")
    ref = load_dataset("parquet", data_files=str(OUT_DIR / "reference_dataset.parquet"), split="train")
    print(f"copied {len(ref)} reference rows -> {OUT_DIR / 'reference_dataset.parquet'} (features={ref.features['image']})")

    # eval uses the combined target set directly (member+nonmember, real true labels via 'label')
    shutil.copy2(RUN1_DIR / "target_dataset.parquet", OUT_DIR / "target_dataset.parquet")
    print(f"copied combined target set -> {OUT_DIR / 'target_dataset.parquet'}")
