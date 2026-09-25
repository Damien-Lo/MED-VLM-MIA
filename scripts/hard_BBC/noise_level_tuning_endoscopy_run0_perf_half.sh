#!/bin/bash
#SBATCH --job-name=endoscopy_run0_noise_tuning_perf_half
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/hard_BBC/out_endoscopy_run0_noise_tuning_perf_half.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Rough/fast tuning pass for endoscopy run_0: every other std value from the original 33-value
# sweep (17 values instead of 33) to roughly halve total wall time, since the full 33-value sweep
# at this pace projects to ~7.6 days -- over the 7-day h200week limit (see prior job 43414,
# cancelled). Same output dir as the cancelled perf run (noise_tuning_perf/) so gn_set0
# (std=0.000, already fully computed, ~12h one-time cost including model load) is reused rather
# than recomputed -- the loop below skips any gn_set whose preds.json already exists.

source ~/.bashrc
conda activate med_vlm_mia_venv

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
reference_dataset='/local/scratch/clo37/datasets/GastroHUN/meta_endoscopy_gastrohun_full.json'
out_dir="${run_dir}/noise_tuning_perf"
descriptions_path="${run_dir}/noise_tuning/gn_set0/datasets/generated_descriptions.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  if [ -f "${out_dir}/gn_set${set}/preds.json" ]; then
    printf "\n>>> gn_set${set} (std=${STD_SETS[$set]}) already complete, skipping.\n\n"
    continue
  fi

  printf "\n>>>===================\n\nENDOSCOPY RUN0 PERF-HALF NOISE TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for endoscopy run_0 (perf branch, reduced metrics+std), std set: ${STD_SETS[$set]}'" \
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
      data.pre_gen_descriptions=${descriptions_path} \
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
