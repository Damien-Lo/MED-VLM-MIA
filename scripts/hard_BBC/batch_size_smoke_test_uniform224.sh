#!/bin/bash
#SBATCH --job-name=batch_size_smoke_test
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/hard_BBC/out_batch_size_smoke_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200hour
#SBATCH --time=01:00:00

# Phase 2 of the uniform-224x224 resize plan: empirically find the largest safe
# inference.batch_size for Hulu-Med-32B now that every image is exactly 256
# vision tokens (uniform size removes the ragged-sequence-length CRASH that
# made batch_size>1 unsafe before). Separately, ALL metrics here force
# augmentation_accumilator=['none'] (not just renyi_divergence_025, the one
# mri_run1_perf.sh patched) -- confirmed 2026-09-21 that the 'avg'/'max'
# cross-setting accumulator in proposed_metrics.py's renyi_kl_div_maxk has a
# SEPARATE bug from the sequence-length crash: even with uniform token length
# (so no shape-mismatch crash), its np.array()-based axis reduction still
# silently collapses multiple samples in a batch into fewer output scores
# (e.g. batch_size=2 -> exactly half as many scores as labels). The 'none'
# path is a correct explicit per-sample loop and unaffected -- and it's the
# only output this whole project's analysis has ever used, so forcing
# ['none'] everywhere costs nothing. Each candidate batch size runs as its own
# subprocess against a tiny 64-sample interleaved (member,nonmember,...) balanced
# dataset -- NOT job_meta_params.test_run=true, which truncates to the first
# batch_size*test_number_of_batches rows; since the full target_dataset.parquet
# is member-block-then-nonmember-block ordered, that truncation would grab an
# all-one-class slice and crash at the final AUC step regardless of batch size
# (confirmed: this is exactly what happened on the first attempt). The 64-sample
# set sidesteps that entirely, and is real job_type=evaluation end-to-end, so a
# clean exit here means the whole pipeline (not just the forward pass) is safe
# at that batch size.

source ~/.bashrc
conda activate med_vlm_mia_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

DATA_DIR='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_09_21_uniform_224x224/endoscopy'
OUT_DIR='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_09_21_uniform_224x224/endoscopy/batch_size_smoke_test'
mkdir -p ${OUT_DIR}

BATCH_SIZES=(1 2 4 8 16 24 32 48 64)

for BS in "${BATCH_SIZES[@]}"; do
  printf "\n>>>===================\n\nBATCH SIZE SMOKE TEST -- batch_size=$BS \n\n=================== \n\n"
  start=$(date +%s)

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Batch-size smoke test on uniform-224x224 endoscopy data, batch_size=${BS}'" \
      job_meta_params.job_type=evaluation \
      \
      path.output_dir=${OUT_DIR}/bs_${BS} \
      \
      target_model=med_hulu \
      target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
      inference.batch_size=${BS} \
      \
      data.save_datasets=false \
      data.target_set_size=64 \
      data.n_nm_ratio=0.5 \
      data.dataset=${DATA_DIR}/datasets/target_dataset_smoketest64.parquet \
      data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_09_21_uniform_224x224/dummy_descriptions_64.json \
      \
      img_metrics.parts=[img] \
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
      data.augmentations.GaussianNoise.std='[0.0078]' \
      data.augmentations.RandomAffine.use=false \
      data.augmentations.ColorJitter.use=false \
      > ${OUT_DIR}/bs_${BS}.log 2>&1
  exit_code=$?

  end=$(date +%s)
  elapsed=$((end - start))

  if [ $exit_code -eq 0 ]; then
    echo "batch_size=${BS}: SUCCESS, wall time ${elapsed}s (see ${OUT_DIR}/bs_${BS}.log for per-batch timing)"
  elif grep -qi "CUDA out of memory\|OutOfMemoryError" ${OUT_DIR}/bs_${BS}.log; then
    echo "batch_size=${BS}: OOM after ${elapsed}s -- stopping sweep here (larger sizes would OOM too)"
    tail -15 ${OUT_DIR}/bs_${BS}.log
    break
  else
    echo "batch_size=${BS}: FAILED for a non-OOM reason (exit code ${exit_code}) after ${elapsed}s -- see ${OUT_DIR}/bs_${BS}.log, continuing to next size"
    tail -30 ${OUT_DIR}/bs_${BS}.log
  fi
done

echo "Smoke test complete."
