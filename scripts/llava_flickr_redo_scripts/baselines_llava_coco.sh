#!/bin/bash
#SBATCH --job-name=baselines_llava_coco_cvpr
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/llava_flickr_redo_scripts/out_baselines_llava_coco.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Blind-binary-classifier-style baselines for LLaVA-v1.5-7B on coco -- mirrors
# baselines_endoscopy_uniform224.sh / baselines_mri_uniform224.sh exactly (job_type=
# evaluation, no Gaussian noise, real generated descriptions, standard baseline metrics
# instead of the proposed noise-perturbation divergence metrics), just target_model +
# dataset swapped. Real descriptions ARE generated here (unlike eval/tuning) since these
# baseline metrics aren't provably independent of description content.
#
# BATCH_SIZE: UNCONFIRMED for llava-v1.5-7b -- real generation is slower per-sample than
# the dummy-description eval/tuning runs, so keep this conservative. See
# tuning_llava_flickr.sh for the same caveat.

source ~/.bashrc
conda activate med_vlm_mia_venv

export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

BATCH_SIZE=4  # TODO: unconfirmed for llava-v1.5-7b with real generation -- smoke-test first

datasets_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/llava-coco/run0/datasets'
out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/MODERN_PIPELINE_RESULTS/llava-coco/run0/baselines'
target_dataset="${datasets_dir}/target_dataset.parquet"

python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Baselines for LLaVA-v1.5-7B members/non-members on coco, CVPR redo'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=${out_dir} \
    \
    target_model="llava-v1.5-7b" \
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
