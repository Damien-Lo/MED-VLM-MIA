#!/bin/bash
#SBATCH --job-name=mri_run1_noise_tuning_perf
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/hard_BBC/out_mri_run1_noise_tuning_perf.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Second trial for mri noise tuning, on a different sampled dataset (run_1, distinct from
# run_0's images -- confirmed via direct parquet inspection) -- for a std across 2 trials of the
# achieved AUC. On perf/vectorize-hulu-batch-and-metrics (batched BatchProcessor_hulu dataset
# access + fast-path support for probabilities/log_probabilities), reduced to the 3 best-
# performing metrics (no_norm, renyi_divergence alpha=4, renyi_divergence alpha=0.25) to engage
# the fast path.
#
# inference.batch_size=2: first attempt at this (job 43563) crashed on every single gn_set with
# a ValueError in renyi_divergence_maxk's np.array(list(all_settings_in_aug.values())) call
# (proposed_metrics.py ~line 347) -- "inhomogeneous shape" because two samples in the same batch
# have different generated-sequence lengths. This is a PRE-EXISTING bug in the original LOGAN
# aggregation code (there's already a `#TODO: ... can't np.array` comment sitting right above it)
# that simply never triggered before because batch_size was always 1, so every call only ever had
# one sample (trivially "homogeneous"). The real fix is to rewrite that aggregation as a per-sample
# loop (stack only same-sample-across-settings, which is always same-length, instead of
# stacking across samples) -- NOT done yet, tracked as follow-up work.
# WORKAROUND applied below instead (time pressure): max_k_renyi_divergence_025 is the only one of
# our 3 metrics with the crash-prone 'max'/'avg' cross-setting accumulator enabled by default
# (max_k_no_norn_kl_div and max_k_renyi_divergence_4 are already 'none'-only). We don't consume
# the 'max'/'avg' aggregated output anywhere (our AUC tables read the per-setting 'none' results,
# computed via a separate ragged-length-safe path), so forcing it to 'none' here just skips the
# broken code path without affecting anything we actually use -- but it means aggregation metrics
# for this metric are unavailable from this run, not fixed.

source ~/.bashrc
conda activate med_vlm_mia_venv

# CUDA's JIT compute cache defaults to $HOME/.nv/ComputeCache, which is NFS -- two
# concurrently-launched Hulu-Med jobs on the same node both hitting that shared NFS
# index file at once wedged both processes in uninterruptible D-state (rpc_wait_bit_
# killable) for 21+ hours on 2026-09-18/19 (jobs 43629/43630). Force it local instead.
export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

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

run_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/mri/run_1'
member_dataset="${run_dir}/datasets/member_target_dataset.parquet"
nonmember_dataset="${run_dir}/datasets/non_member_target_dataset.parquet"
reference_dataset='/local/scratch/clo37/datasets/TCIA/combined_datasets/05_brain_mri_05_other_mri.json'
out_dir="${run_dir}/noise_tuning"
# Dummy descriptions (600x "N/A"): img_metrics.parts=["img"] only scores image-token logits, which
# causal attention makes provably independent of any text that comes after the images in the
# conversation (inst/desc). Skips the "Generating responses" step entirely -- also the exact step
# that hung on NFS earlier.
pre_gen_arg="data.pre_gen_descriptions=${run_dir}/dummy_descriptions.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nMRI RUN1 PERF NOISE TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for mri run_1 (perf branch, reduced metrics), std set: ${STD_SETS[$set]}'" \
      job_meta_params.job_type=hyperparam_tuning \
      \
      path.output_dir=${out_dir}/gn_set${set} \
      \
      target_model="med_hulu" \
      target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
      inference.batch_size=2 \
      \
      data.save_datasets=true \
      data.target_set_size=300 \
      data.n_nm_ratio=0.5 \
      data.member_dataset=${member_dataset} \
      data.nonmember_dataset=${nonmember_dataset} \
      data.reference_datasets_list=${reference_dataset} \
      data.reference_set_sample_distribution=[] \
      ${pre_gen_arg} \
      \
      img_metrics.parts=["img"] \
      img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_divergence_4","max_k_renyi_divergence_025"]' \
      img_metrics.get_raw_meta_metrics=[] \
      img_metrics.get_proc_meta_metrics=[] \
      img_metrics.max_k_renyi_divergence_025.augmentation_accumilator='["none"]' \
      img_metrics.max_k_renyi_divergence_025.augmentation_setting_version_accumilator='["none"]' \
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
