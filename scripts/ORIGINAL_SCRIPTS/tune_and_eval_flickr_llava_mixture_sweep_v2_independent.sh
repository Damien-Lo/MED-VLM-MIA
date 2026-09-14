#!/bin/bash
#SBATCH --job-name=flickr_llava_mixture_sweep_v2
#SBATCH --output=out_flickr_llava_mixture_sweep_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Corrected rerun of tune_and_eval_flickr_llava_mixture_sweep.sh -- see
# additional_ICLR_experiments/KNOWN_ISSUE_exact_nonmember_reference.md for the full writeup.
# The original sweep built each mixture's "target-like" reference component from the target
# set's own *exact* non-member images, which (algebraically) makes the gap's non-member
# coefficient (0.5-p) cross zero at p=0.5 and flip sign beyond it -- not genuine damping.
#
# Fix: reference sets for p=0..90% now use build_mixture_reference_sets_v2_independent.py's
# output (an *independently sampled*, disjoint-from-target flickr non-member pool mixed with
# ShareGPT), matching how the original Figure 3 ("Estimated" = sharegpt0.5_flickr0.5_KN) was
# actually built. p=100% is unchanged -- reuses the ORIGINAL reference_dataset_p100.parquet
# (mixture_reference_sets/, not v2_independent/) directly, since that endpoint was always meant
# to be the exact target non-members and was never affected by the bug.
#
# p=0.5 runs FIRST in this sweep (not p=0.0) specifically so its tuning_separation.json can be
# plotted against the true-optimal member-vs-nonmember curve early, as a sanity check that the
# fix actually restores sign agreement, before committing to the full ~4-5 day sweep.
#
# Everything else (std grid, metrics, model, prompt, k_ratio=1.0, one-std-per-job,
# generate-descriptions-once-per-mixture-then-reuse) is unchanged from the original script.

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

# p=050 first (sanity check against true optimal), then the rest of the sweep in order.
MIXTURE_PS=(050 000 010 020 030 040 060 070 080 090 100)

iclr_experiments_dir='/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/additional_ICLR_experiments'
target_dataset="${iclr_experiments_dir}/datasets/target_dataset.parquet"
mixture_ref_dir_v2="${iclr_experiments_dir}/datasets/mixture_reference_sets_v2_independent"
mixture_ref_dir_orig="${iclr_experiments_dir}/datasets/mixture_reference_sets"
out_root="${iclr_experiments_dir}/results_independent_flickr_pool"


for pp in "${MIXTURE_PS[@]}"; do
  if [ "$pp" == "100" ]; then
    # Unchanged endpoint -- reuse the original (exact target non-members) reference set as-is.
    reference_dataset="${mixture_ref_dir_orig}/reference_dataset_p100.parquet"
  else
    reference_dataset="${mixture_ref_dir_v2}/reference_dataset_p${pp}.parquet"
  fi
  out_dir="${out_root}/mix_p${pp}"
  descriptions_path="${out_dir}/gn_set0/datasets/generated_descriptions.json"

  for ((set=0; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nMIXTURE p=0.%s (independent-pool v2) -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n" "${pp}"

    pre_gen_arg=""
    if [ "$set" -ne 0 ]; then
      pre_gen_arg="data.pre_gen_descriptions=${descriptions_path}"
    fi

    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Flickr/LLaVA mixture-sweep rerun (independent flickr pool v2): reference = ${pp}% independently-sampled flickr non-members / $((100 - 10#$pp))% ShareGPT, std set: ${STD_SETS[$set]}'" \
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
