#!/bin/bash
#SBATCH --job-name=endoscopy_run1_noise_tuning_perf_fixedref
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/hard_BBC/out_endoscopy_run1_noise_tuning_perf_fixedref.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Clean restart of endoscopy run_1's halved-std tuning -- see
# noise_level_tuning_endoscopy_run0_perf_fixedref.sh for the full rationale (frozen 300-image
# reference set instead of resampling every gn_set, resized to 640x480 to match the member set's
# median resolution instead of native 1350x1080).
#
# New output dir (noise_tuning_fixedref/, not noise_tuning/) since the old gn_set0 used the old
# resampled-native-res reference and is no longer consistent with this run. gn_set0 here
# regenerates its own descriptions fresh so they're based on the same fixed+shrunk reference
# images actually used for inference.

source ~/.bashrc
conda activate med_vlm_mia_venv

# CUDA's JIT compute cache defaults to $HOME/.nv/ComputeCache, which is NFS -- two
# concurrently-launched Hulu-Med jobs on the same node both hitting that shared NFS
# index file at once wedged both processes in uninterruptible D-state (rpc_wait_bit_
# killable) for 21+ hours on 2026-09-18/19 (jobs 43629/43630). Force it local instead.
export CUDA_CACHE_PATH=/local/scratch/clo37/cache/nv_compute_cache

STD_SETS=(
  "[0.000]" "[0.0078]" "[0.019]"
  "[0.046]" "[0.11]" "[0.28]"
  "[0.67]" "[1.6]" "[4.0]"
  "[9.8]" "[24]" "[58]"
  "[140]" "[340]" "[840]"
  "[2100]" "[5000]"
)

run_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/endoscopy/run_1'
member_dataset="${run_dir}/datasets/member_target_dataset.parquet"
nonmember_dataset="${run_dir}/datasets/non_member_target_dataset.parquet"
reference_dataset="${run_dir}/fixed_reference_640x480/reference_dataset_fixed_640x480.parquet"
out_dir="${run_dir}/noise_tuning_fixedref"
# Dummy descriptions (600x "N/A") -- see run0_perf_fixedref.sh for why: img_metrics.parts=["img"]
# only scores image-token logits, which causal attention makes provably independent of any text
# that comes after the images in the conversation (inst/desc). Skips the "Generating responses"
# step entirely -- also the exact step that hung on NFS earlier.
pre_gen_arg="data.pre_gen_descriptions=${run_dir}/fixed_reference_640x480/dummy_descriptions.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nENDOSCOPY RUN1 PERF-FIXEDREF NOISE TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for endoscopy run_1 (perf branch, reduced metrics+std, fixed+shrunk reference), std set: ${STD_SETS[$set]}'" \
      job_meta_params.job_type=hyperparam_tuning \
      \
      path.output_dir=${out_dir}/gn_set${set} \
      \
      target_model="med_hulu" \
      target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
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
      img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_divergence_4","max_k_renyi_divergence_025"]' \
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
