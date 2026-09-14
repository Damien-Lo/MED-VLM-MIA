#!/bin/bash
#SBATCH --job-name=flickr_llava_mixture_sweep
#SBATCH --output=out_flickr_llava_mixture_sweep.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week


# Reruns the old LLaVA/flickr hyperparam_tuning + eval (originally under
# .../FINAL_RESULTS/llava/flickr/hyperparam_tuning/flickr_pretrain_member_ratio_0.5__sharegpt_KN
# -- target/member/non-member/reference parquet copies now live under
# additional_ICLR_experiments/datasets/, see build_mixture_reference_sets.py) using the new
# job_type=tune_and_eval, once per reference-set mixture fraction -- 11 reference sets
# interpolating from pure ShareGPT (p=0.0) to the target set's own 150 flickr non-members
# (p=1.0), so the resulting tuning_separation.json files can be overlaid into an 11-curve
# version of the Delta-Renyi-divergence-vs-sigma figure.
#
# Only the two requested metrics (no normalization / Renyi alpha=inf normalization) are run to
# keep the 11 mixtures x 33 std points x model-load cost manageable. One std per job (matches
# every other hard_BBC-derived script here) -- OOM was hit in the past running multiple stds
# per job.
#
# Reuses the *original* target_dataset.parquet (300 rows: 150 flickr members + 150 flickr
# non-members) unchanged across every mixture/std -- only the reference set varies. This is a
# flickr/LLaVA/ICLR-scoped rerun, not a hard_BBC (medical) experiment -- see
# scripts/ORIGINAL_SCRIPTS/ vs scripts/hard_BBC/.

# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv


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

# 0%, 10%, ..., 100% -- must match build_mixture_reference_sets.py's MIXTURE_FRACTIONS
MIXTURE_PS=(000 010 020 030 040 050 060 070 080 090 100)

iclr_experiments_dir='/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/additional_ICLR_experiments'
target_dataset="${iclr_experiments_dir}/datasets/target_dataset.parquet"
mixture_ref_dir="${iclr_experiments_dir}/datasets/mixture_reference_sets"
out_root="${iclr_experiments_dir}/results"  # one mix_p${pp}/ subdir per reference-set mixture, gn_set0..32 (one per std) inside each


for pp in "${MIXTURE_PS[@]}"; do
  reference_dataset="${mixture_ref_dir}/reference_dataset_p${pp}.parquet"
  out_dir="${out_root}/mix_p${pp}"
  descriptions_path="${out_dir}/gn_set0/datasets/generated_descriptions.json"

  for ((set=0; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nMIXTURE p=0.%s -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n" "${pp}"

    # Descriptions only depend on (model, prompt, target+reference images) -- generate once per
    # mixture on the first std set, then reuse across the other 32 std sets for that mixture.
    pre_gen_arg=""
    if [ "$set" -ne 0 ]; then
      pre_gen_arg="data.pre_gen_descriptions=${descriptions_path}"
    fi

    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Flickr/LLaVA mixture-sweep rerun: reference = ${pp}% flickr target non-members / $((100 - 10#$pp))% ShareGPT, std set: ${STD_SETS[$set]}'" \
        job_meta_params.job_type=tune_and_eval \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model="llava-v1.5-7b" \
        target_model.model_path='liuhaotian/llava-v1.5-7b' \
        prompt.text="'Describe this image concisely.'" \
        \
        data.save_datasets=false \
        data.dataset=${target_dataset} \
        data.reference_dataset=${reference_dataset} \
        ${pre_gen_arg} \
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
        data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false
  done
done
