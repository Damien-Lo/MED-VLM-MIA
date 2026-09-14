#!/bin/bash
#SBATCH --job-name=smoke_test_mixture_sweep
#SBATCH --output=out_smoke_test_mixture_sweep.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200hour

# One-off smoke test for tune_and_eval_flickr_llava_mixture_sweep.sh -- confirms the sentencepiece
# fix (2026-08-31) actually lets a real job load LLaVA and run to completion, before relaunching
# the full 363-job sweep. test_run=true caps it at 5 batches. Single combo: mix_p000, std=0.005
# (skips std=0.000 since that's a degenerate no-noise case). Writes to a throwaway smoke_test/ dir,
# not into results/, so it can't collide with or contaminate the real sweep output.

source ~/.bashrc
conda activate med_vlm_mia_venv

iclr_experiments_dir='/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/additional_ICLR_experiments'
target_dataset="${iclr_experiments_dir}/datasets/target_dataset.parquet"
reference_dataset="${iclr_experiments_dir}/datasets/mixture_reference_sets/reference_dataset_p000.parquet"
out_dir="${iclr_experiments_dir}/results_smoke_test/mix_p000_std0.005"

python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=true \
    job_meta_params.description="'Smoke test: sentencepiece fix verification'" \
    job_meta_params.job_type=tune_and_eval \
    inference.test_number_of_batches=160 \
    \
    path.output_dir=${out_dir} \
    \
    target_model="llava-v1.5-7b" \
    target_model.model_path='liuhaotian/llava-v1.5-7b' \
    prompt.text="'Describe this image concisely.'" \
    \
    data.save_datasets=false \
    data.dataset=${target_dataset} \
    data.reference_dataset=${reference_dataset} \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_inf_kl_div"]' \
    img_metrics.get_raw_meta_metrics=[] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.max_k_no_norn_kl_div.ratio='[1.0]' \
    img_metrics.max_k_renyi_inf_kl_div.ratio='[1.0]' \
    \
    img_metrics.get_meta_examples=1000 \
    img_metrics.get_token_labels=0 \
    img_metrics.get_raw_images=0 \
    \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomRotation.use=false \
    data.augmentations.GaussianNoise.use=true \
    data.augmentations.GaussianNoise.mean='[0.0]' \
    data.augmentations.GaussianNoise.std='[0.005]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false
