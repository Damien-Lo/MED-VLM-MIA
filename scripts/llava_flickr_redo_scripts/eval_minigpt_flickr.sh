#!/bin/bash
#SBATCH --job-name=eval_minigpt_flickr_cvpr
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/llava_flickr_redo_scripts/out_eval_minigpt_flickr.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# *** DO NOT SUBMIT AS-IS ***
# vlm_mia_llava_minigpt_venv is currently BROKEN (confirmed 2026-09-23): missing basic
# deps -- `python3 -c "import transformers"` fails with `ModuleNotFoundError: No module
# named 'requests'`. Fix the env (or rebuild it) and re-verify `import torch, transformers`
# succeeds before submitting this job.

# Full noise-sweep evaluation (job_type=evaluation, real member-vs-non-member AUC, all 9
# metrics) for MiniGPT-4 on flickr, fresh inference -- mirrors eval_endoscopy_uniform224.sh
# / eval_mri_uniform224.sh exactly, same job_type/std-grid/metrics structure, just
# target_model + dataset swapped.
#
# target dataset: the combined 300-row target set (150 member/150 non-member, real 'label'
# column) built by build_cvpr_datasets.py from the archive's run_1/gn_set0 data.
#
# Dummy descriptions ("N/A" x300): same img-only rationale as eval_endoscopy_uniform224.sh.
#
# All metrics force augmentation_accumilator=['none'] -- same known aggregation bug as
# documented in eval_endoscopy_uniform224.sh.
#
# BATCH_SIZE: UNCONFIRMED for minigpt-4 at this scale -- see tuning_llava_flickr.sh.

source ~/.bashrc
conda activate vlm_mia_llava_minigpt_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

BATCH_SIZE=16  # TODO: unconfirmed at scale for minigpt-4 -- smoke-test before relying on this

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

datasets_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/minigpt-flickr/run0/datasets'
out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/minigpt-flickr/run0/eval'
target_dataset="${datasets_dir}/target_dataset.parquet"
pre_gen_arg="data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/llava_low_tpr_redo/dummy_descriptions_300.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nMINIGPT FLICKR CVPR EVAL -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Full-sweep evaluation for MiniGPT-4 on flickr, CVPR redo, std set: ${STD_SETS[$set]}'" \
      job_meta_params.job_type=evaluation \
      \
      path.output_dir=${out_dir}/gn_set${set} \
      \
      target_model="minigpt-4" \
      target_model.model.ckpt='/local/scratch/clo37/models/minigpt4/pretrained_minigpt4_llama2_7b.pth' \
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
