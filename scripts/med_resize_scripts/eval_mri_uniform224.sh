#!/bin/bash
#SBATCH --job-name=eval_mri_uniform224
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/med_resize_scripts/out_eval_mri_uniform224.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Full noise-sweep evaluation (job_type=evaluation, real member-vs-non-member AUC,
# all 9 metrics) for mri, on the uniform-224x224-resized dataset built in
# results/2026_09_21_uniform_224x224 -- every image is exactly 256 vision tokens,
# so batch_size>1 is safe (no more ragged-sequence-length crash on renyi_maxk).
# This also removes the resolution confound found in the original data: real
# ROCOv2 members ranged 46,161-2,139,750 pixels while synthetic Prompt2MedImage
# non-members were a fixed 512x512 (262,144 px) -- resolution alone could have
# been a distinguishing feature independent of genuine membership signal.
#
# Dummy descriptions ("N/A" x300): img_metrics.parts=["img"] only scores the
# image-token slice, provably independent of description content under causal
# attention -- see eval_endoscopy_uniform224.sh for the full rationale.
#
# All metrics force augmentation_accumilator=['none'] -- confirmed 2026-09-21 that
# the 'avg'/'max' cross-setting accumulator has a bug separate from the sequence-
# length crash the uniform resize fixes: it silently collapses multiple samples in
# a batch into fewer scores at batch_size>1 even with uniform token length. The
# 'none' path is unaffected and is the only output this project's analysis uses.
#
# BATCH_SIZE: set from the Phase-2 smoke test (batch_size_smoke_test_uniform224.sh)
# -- update this once that result lands.

source ~/.bashrc
conda activate med_vlm_mia_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

BATCH_SIZE=32  # confirmed safe up to 48 in Phase-2 smoke test (job 44340), no OOM observed

STD_SETS=(
  "[0.000]" "[0.005]" "[0.0078]"
  "[0.012]" "[0.019]" "[0.03]"
  "[0.046]" "[0.072]" "[0.11]"
  "[0.18]" "[0.28]" "[0.43]"
  "[0.67]" "[1.1]" "[1.6]"
  "[2.6]" "[4.0]" "[6.2]"
  "[9.8]" "[15]" "[24]"
  "[37]" "[58]" "[91]"
  "[140]" "[220]" "[340]"
  "[540]" "[840]" "[1300]"
  "[2100]" "[3200]" "[5000]"
)

data_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_09_21_uniform_224x224/mri'
out_dir="${data_dir}/eval"
target_dataset="${data_dir}/datasets/target_dataset.parquet"
pre_gen_arg="data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_09_21_uniform_224x224/dummy_descriptions_300.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nMRI UNIFORM224 EVAL -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Full-sweep evaluation for mri, uniform 224x224 resize (all images 256 vision tokens), std set: ${STD_SETS[$set]}'" \
      job_meta_params.job_type=evaluation \
      \
      path.output_dir=${out_dir}/gn_set${set} \
      \
      target_model="med_hulu" \
      target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
      inference.batch_size=${BATCH_SIZE} \
      \
      data.save_datasets=true \
      data.target_set_size=300 \
      data.n_nm_ratio=0.5 \
      data.dataset=${target_dataset} \
      ${pre_gen_arg} \
      \
      img_metrics.parts=["img"] \
      img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"]' \
      img_metrics.max_k_no_norn_kl_div.augmentation_accumilator=['none'] \
      img_metrics.max_k_no_norn_kl_div.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_05_kl_div.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_05_kl_div.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_1_kl_div.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_1_kl_div.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_2_kl_div.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_2_kl_div.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_inf_kl_div.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_inf_kl_div.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_025.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_025.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_05.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_05.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_2.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_2.augmentation_setting_version_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_4.augmentation_accumilator=['none'] \
      img_metrics.max_k_renyi_divergence_4.augmentation_setting_version_accumilator=['none'] \
      img_metrics.get_raw_meta_metrics=[] \
      img_metrics.get_proc_meta_metrics=[] \
      \
      img_metrics.get_meta_examples=1000 \
      img_metrics.get_token_labels=1000 \
      img_metrics.get_raw_images=0 \
      \
      data.augmentations.RandomResize.use=false \
      data.augmentations.RandomRotation.use=false \
      data.augmentations.GaussianNoise.use=true \
      data.augmentations.GaussianNoise.mean='[0.0]' \
      data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
      data.augmentations.RandomAffine.use=false \
      data.augmentations.ColorJitter.use=false
done
