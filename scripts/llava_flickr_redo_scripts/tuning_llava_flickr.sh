#!/bin/bash
#SBATCH --job-name=tuning_llava_flickr_cvpr
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/llava_flickr_redo_scripts/out_tuning_llava_flickr.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Full noise-sweep hyperparam_tuning (target-set vs. reference-set divergence, all 9
# metrics) for LLaVA-v1.5-7B on flickr, fresh inference through the current/modern
# pipeline -- mirrors tuning_endoscopy_uniform224.sh / tuning_mri_uniform224.sh exactly,
# same job_type/std-grid/metrics structure, just target_model + dataset swapped.
#
# member/non-member/reference datasets: built by build_cvpr_datasets.py from the archive's
# own run_1/gn_set0 data -- the one genuinely-valid 50/50 flickr(non-member-distribution)/
# ShareGPT mixture-reference tuning run (150 flickr member/150 flickr non-member target set,
# 150 flickr-distribution + 150 ShareGPT reference set). Confirmed via
# scripts/ORIGINAL_SCRIPTS/build_mixture_reference_sets_v4_run1_anchored.py's docstring that
# this exact run_1 data reproduces the published Figure 3 at p=0.5.
#
# Dummy descriptions ("N/A" x600, target+reference): img_metrics.parts=["img"] only scores
# the image-token slice of the logits, description content is provably irrelevant -- same
# rationale as eval_endoscopy_uniform224.sh. Reusing the exact files already built+verified
# this session (300/300 row counts match) at llava_low_tpr_redo/dummy_descriptions_*.json.
#
# All metrics force augmentation_accumilator=['none'] -- same known 'avg'/'max' aggregation
# bug as documented in tuning_endoscopy_uniform224.sh.
#
# BATCH_SIZE: UNCONFIRMED for llava-v1.5-7b at this scale -- set conservatively. LLaVA-7B is
# far smaller than Hulu-Med-32B (which was confirmed safe at 32), so this can very likely go
# higher, but that hasn't been smoke-tested. Run a batch-size smoke test before trusting
# throughput here; lower if OOM.

source ~/.bashrc
conda activate med_vlm_mia_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

BATCH_SIZE=16  # TODO: unconfirmed at scale for llava-v1.5-7b -- smoke-test before relying on this

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

datasets_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/llava-flickr/run0/datasets'
out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/llava-flickr/run0/tuning'
member_dataset="${datasets_dir}/member_target_dataset.parquet"
nonmember_dataset="${datasets_dir}/non_member_target_dataset.parquet"
reference_dataset="${datasets_dir}/reference_dataset.parquet"
pre_gen_arg="data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/llava_low_tpr_redo/dummy_descriptions_600.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nLLAVA FLICKR CVPR TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for LLaVA-v1.5-7B on flickr, CVPR redo, 50/50 mixture reference (run_1-anchored), std set: ${STD_SETS[$set]}'" \
      job_meta_params.job_type=hyperparam_tuning \
      \
      path.output_dir=${out_dir}/gn_set${set} \
      \
      target_model="llava-v1.5-7b" \
      inference.batch_size=${BATCH_SIZE} \
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
