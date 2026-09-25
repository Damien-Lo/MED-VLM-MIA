#!/bin/bash
#SBATCH --job-name=baselines_mri_uniform224
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/med_resize_scripts/out_baselines_mri_uniform224.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Blind-binary-classifier-style baselines for mri on the uniform-224x224
# dataset -- mirrors baselines_32B.sh's methodology exactly (job_type=evaluation,
# no Gaussian noise, real generated descriptions, standard baseline metrics
# instead of the proposed noise-perturbation divergence metrics), just pointed
# at the resized dataset.
#
# BATCH_SIZE: set from the Phase-2 smoke test (batch_size_smoke_test_uniform224.sh,
# job 44320) -- update this once that result lands. Note baselines use real
# generation (slower per-sample than the dummy-description eval/tuning runs),
# so the safe batch size found there may not transfer exactly -- sanity check
# if this OOMs.

source ~/.bashrc
conda activate med_vlm_mia_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

BATCH_SIZE=8  # TODO: update from smoke test 44320's result before submitting for real

data_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_09_21_uniform_224x224/mri'
out_dir="${data_dir}/baselines"
target_dataset="${data_dir}/datasets/target_dataset.parquet"

python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Baselines for mri members/non-members, uniform 224x224 resize'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=${out_dir} \
    \
    target_model="med_hulu" \
    target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
    inference.batch_size=${BATCH_SIZE} \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions="" \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=['aug_kl','max_k_renyi_1_entro','max_k_renyi_05_entro','mink'] \
    img_metrics.get_raw_meta_metrics=[] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.get_meta_examples=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=0 \
    \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomRotation.use=true \
    data.augmentations.GaussianNoise.use=false \
    data.augmentations.RandomAffine.use=true \
    data.augmentations.ColorJitter.use=true
