#!/bin/bash
#SBATCH --job-name=flickr_llava_mixture_sweep_v3
#SBATCH --output=out_flickr_llava_mixture_sweep_v3.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Third attempt at the mixture sweep -- see
# /local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results/README.txt for the full
# history. Uses build_mixture_reference_sets_v3_unrestricted.py's output: p=0/50/100 are direct
# reuses of real/validated files (p=50 is byte-identical to the data that reproduced the
# published Figure 3 exactly), p=10..40/60..90 use unrestricted sampling from the real flickr
# non-member pool (fixing v2's artificial exclusion of the target's own non-members).
#
# p=050 runs FIRST as the sanity check (near-certain to reproduce correctly, since it's real
# validated data, not a new construction), then the rest of the sweep continues.

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

# p=050 first (sanity check against real reproduction), then the rest of the sweep in order.
MIXTURE_PS=(050 000 010 020 030 040 060 070 080 090 100)

iclr_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results'
target_dataset="${iclr_dir}/datasets/target_dataset.parquet"
mixture_ref_dir_v3="${iclr_dir}/datasets/mixture_reference_sets_v3_unrestricted"
out_root="${iclr_dir}/v3_unrestricted_pool/results"


for pp in "${MIXTURE_PS[@]}"; do
  reference_dataset="${mixture_ref_dir_v3}/reference_dataset_p${pp}.parquet"
  out_dir="${out_root}/mix_p${pp}"
  descriptions_path="${out_dir}/gn_set0/datasets/generated_descriptions.json"

  for ((set=0; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nMIXTURE p=0.%s (v3 unrestricted pool) -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n" "${pp}"

    pre_gen_arg=""
    if [ "$set" -ne 0 ]; then
      pre_gen_arg="data.pre_gen_descriptions=${descriptions_path}"
    fi

    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Flickr/LLaVA mixture-sweep rerun (v3, unrestricted pool): reference = ${pp}% flickr non-members (unrestricted pool) / $((100 - 10#$pp))% ShareGPT, std set: ${STD_SETS[$set]}'" \
        job_meta_params.job_type=tune_and_eval \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model="llava-v1.5-7b" \
        target_model.model_path='liuhaotian/llava-v1.5-7b' \
        inference.batch_size=4 \
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
