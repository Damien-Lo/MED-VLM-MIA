#!/bin/bash
#SBATCH --job-name=smoke_test_llava_flickr
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/llava_flickr_redo_scripts/out_smoke_test_llava_flickr.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200hour

# Phase 1 smoke test: does the current (perf worktree) pipeline actually run
# target_model=llava-v1.5-7b end to end under med_vlm_mia_venv? test_run=true,
# batch_size=1, 1 std point only. Not a batch-size sweep yet -- just "does it work".

source ~/.bashrc
conda activate med_vlm_mia_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

data_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/llava_low_tpr_redo'
member_dataset="${data_dir}/datasets/member_target_dataset.parquet"
nonmember_dataset="${data_dir}/datasets/non_member_target_dataset.parquet"
reference_dataset="${data_dir}/datasets/reference_dataset.parquet"

python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
    job_meta_params.test_run=true \
    inference.test_number_of_batches=1 \
    job_meta_params.description="'Smoke test: llava-v1.5-7b on flickr tuning pipeline'" \
    job_meta_params.job_type=hyperparam_tuning \
    \
    path.output_dir=${data_dir}/smoke_test \
    \
    target_model="llava-v1.5-7b" \
    inference.batch_size=1 \
    \
    data.save_datasets=false \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.member_dataset=${member_dataset} \
    data.nonmember_dataset=${nonmember_dataset} \
    data.reference_datasets_list=${reference_dataset} \
    data.reference_set_sample_distribution=[] \
    data.pre_gen_descriptions=${data_dir}/dummy_descriptions_600.json \
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
    data.augmentations.GaussianNoise.std='[37]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false
