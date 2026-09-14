#!/bin/bash
#SBATCH --job-name=flickr_llava_mixture_sweep_v4_resume
#SBATCH --output=out_flickr_llava_mixture_sweep_v4_resume.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Resume of tune_and_eval_flickr_llava_mixture_sweep_v4_run1_anchored.sh (job 42340), killed and
# restarted to raise GPU utilization -- cluster monitoring flagged 42340 for near-0% GPU compute
# and low VRAM use (14GB/143GB) despite steady progress; root cause is that noise augmentation
# (AddGaussianNoisePIL) and image preprocessing are CPU/PIL-only by design, so the GPU sat mostly
# idle between short forward-pass bursts. Fix here: inference.batch_size 4->16 (plenty of VRAM
# headroom) and cpus-per-task 8->16 (more parallelism for the CPU-bound augmentation step); no
# change to any experiment logic (noise application, metrics, datasets).
#
# Picks up exactly where 42340 left off:
#   p=050,000,010,020,030: already fully complete (33/33 gn_sets each) -- skipped entirely.
#   p=040: complete through gn_set25; gn_set26 was mid-write when killed (only pre-inference files
#          present, no tuning_raw_scores.json/preds.json) -- redone from gn_set26 through gn_set32.
#   p=060,070,080,090,100: untouched -- run in full (gn_set0..32).

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

# Remaining work only: p=040 resumes at gn_set26, then 060/070/080/090/100 run in full.
MIXTURE_PS=(040 060 070 080 090 100)

iclr_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results'
target_dataset="${iclr_dir}/datasets/target_dataset_run1_exact.parquet"
mixture_ref_dir_v4="${iclr_dir}/datasets/mixture_reference_sets_v4_run1_anchored"
out_root="${iclr_dir}/v4_run1_anchored/results"


for pp in "${MIXTURE_PS[@]}"; do
  reference_dataset="${mixture_ref_dir_v4}/reference_dataset_p${pp}.parquet"
  out_dir="${out_root}/mix_p${pp}"
  descriptions_path="${out_dir}/gn_set0/datasets/generated_descriptions.json"

  if [ "$pp" == "040" ]; then
    start_set=26
  else
    start_set=0
  fi

  for ((set=start_set; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nMIXTURE p=0.%s (v4 run_1-anchored, resumed) -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n" "${pp}"

    # gn_set0's descriptions are always needed as the pre-gen source (generated once, reused for
    # every other std) -- for p=040 that's already on disk from before the kill, for the fresh
    # p values gn_set0 runs first in this loop and generates it.
    pre_gen_arg=""
    if [ "$set" -ne 0 ]; then
      pre_gen_arg="data.pre_gen_descriptions=${descriptions_path}"
    fi

    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Flickr/LLaVA mixture-sweep rerun (v4, run_1-anchored, resumed at higher batch size): reference = ${pp}% flickr non-members (run_1-anchored) / $((100 - 10#$pp))% ShareGPT, std set: ${STD_SETS[$set]}'" \
        job_meta_params.job_type=tune_and_eval \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model="llava-v1.5-7b" \
        target_model.model_path='liuhaotian/llava-v1.5-7b' \
        inference.batch_size=16 \
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
