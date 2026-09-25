#!/bin/bash
#SBATCH --job-name=endoscopy_run0_noise_tuning_perf_fixedref
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/hard_BBC/out_endoscopy_run0_noise_tuning_perf_fixedref.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Clean restart of endoscopy run_0's halved-std tuning (superseding noise_level_tuning_
# endoscopy_run0_perf_half.sh, killed after 2/17 gn_sets with two problems found:
#   1) The reference set was being RESAMPLED from the full 8834-image GastroHUN pool on every
#      single gn_set (confirmed: gn_set0 and gn_set1 used two entirely different 300-image draws)
#      instead of being held fixed across the noise sweep like member/nonmember target sets are.
#      That's a real methodology bug -- the divergence-vs-noise-level comparison needs the same
#      reference baseline at every noise level, not a different random draw each time.
#   2) The reference images (GastroHUN, uniform 1350x1080) are ~4.4x more pixels than the target
#      member set's median resolution (666x521) -- since Hulu-Med is a dynamic-resolution model
#      (vision tokens scale with input pixel count), this gave reference images disproportionately
#      more tokens than the images they're meant to be a baseline against, and cost real wall time.
#
# Fix: data.reference_datasets_list now points at a FROZEN, pre-shrunk parquet
# (fixed_reference_640x480/reference_dataset_fixed_640x480.parquet) built from gn_set0's own
# original 300-image reference draw (same images, just resized once to 640x480, close to the
# member set's median scale) -- same 300 images, same resolution, every gn_set.
#
# New output dir (noise_tuning_perf_fixedref/, not noise_tuning_perf/) since the old gn_set0/1
# used the old resampled-native-res reference and are no longer consistent with this run.
# gn_set0 here regenerates its own descriptions fresh (not reusing the old run's
# generated_descriptions.json) so the pre-generated description text is based on the same
# fixed+shrunk reference images actually used for inference, not the old native-res ones.

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

run_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/endoscopy/run_0'
member_dataset="${run_dir}/datasets/member_target_dataset.parquet"
nonmember_dataset="${run_dir}/datasets/non_member_target_dataset.parquet"
reference_dataset="${run_dir}/fixed_reference_640x480/reference_dataset_fixed_640x480.parquet"
out_dir="${run_dir}/noise_tuning_perf_fixedref"
# Dummy descriptions (600x "N/A"): img_metrics.parts=["img"] only scores the image-token slice of
# the logits, and the conversation puts images before inst/desc text -- under causal attention the
# image-token logits can never be influenced by text that comes later in the sequence. So the
# actual description content is provably irrelevant to any metric we compute, and generating real
# ones (the "Generating responses" step) is pure wasted cost -- it's also the exact step that hung
# on NFS earlier. Skips generation entirely via the existing pre_gen_descriptions load path.
pre_gen_arg="data.pre_gen_descriptions=${run_dir}/fixed_reference_640x480/dummy_descriptions.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nENDOSCOPY RUN0 PERF-FIXEDREF NOISE TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for endoscopy run_0 (perf branch, reduced metrics+std, fixed+shrunk reference), std set: ${STD_SETS[$set]}'" \
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
