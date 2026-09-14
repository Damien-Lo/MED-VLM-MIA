#!/bin/bash
#SBATCH --job-name=endoscopy_noise_tuning
#SBATCH --output=out_endoscopy_noise_tuning.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week


# Hyperparameter tuning (job_type=hyperparam_tuning) for the endoscopy hard-BBC-filtered MIA run
# -- i.e. computing the target-set-vs-reference-set divergence across the full noise range, the
# step the published run_0 results skipped (run_0 only ran job_type=evaluation). Reuses the
# *exact* 300 samples (150 members + 150 non-members) run_0's real eval used, by loading its
# already-frozen member_target_dataset.parquet / non_member_target_dataset.parquet directly
# (build_hyperparam_full_set's .parquet branch concatenates them as-is, no resampling) -- so
# these tuning results are directly comparable to run_0's published AUC curves.
#
# Reference set: GastroHUN (Hospital Universitario Nacional de Colombia, Bogota -- real upper-GI
# videoendoscopy, see GastroHUN/preprocessing.py), genuinely disjoint from both PubMedVision (the
# member source) and all 21 datasets folded into EndoBench (the non-member source: Kvasir,
# HyperKvasir, GastroVision, SUN, Cholec80, etc. -- verified via EndoBench.json's per-sample
# `dataset` field that the current non-member pool already draws from every one of them, so none
# of those would actually be an independent reference set).
#
# One std per job (matches every other hard_BBC script here) -- OOM was hit in the past running
# multiple stds per job. Descriptions only depend on (model, prompt, target+reference images),
# not augmentation, so they're generated once on gn_set0 and reused for gn_set1..32 via
# data.pre_gen_descriptions -- needed fresh here (can't reuse run_0's baseline descriptions)
# since the reference-set rows never had descriptions generated before.
#
# Results land under a new noise_tuning/ directory alongside run_0's existing gn_set*/ eval
# output, not mixed into it.

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

run_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/endoscopy/run_0'
member_dataset="${run_dir}/datasets/member_target_dataset.parquet"
nonmember_dataset="${run_dir}/datasets/non_member_target_dataset.parquet"
reference_dataset='/local/scratch/clo37/datasets/GastroHUN/meta_endoscopy_gastrohun_full.json'
out_dir="${run_dir}/noise_tuning"
descriptions_path="${out_dir}/gn_set0/datasets/generated_descriptions.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nENDOSCOPY NOISE TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  pre_gen_arg=""
  if [ "$set" -ne 0 ]; then
    pre_gen_arg="data.pre_gen_descriptions=${descriptions_path}"
  fi

  python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for endoscopy (hard BBC filtered target set vs. GastroHUN out-of-distribution reference), std set: ${STD_SETS[$set]}'" \
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
      img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"]' \
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
