#!/bin/bash
#SBATCH --job-name=endoscopy_run0_noise_tuning_perf
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/scripts/hard_BBC/out_endoscopy_run0_noise_tuning_perf.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Redo of endoscopy/run_0's noise tuning (job 41691, TIMEOUT'd at gn_set6/32 after 7 days) --
# on perf/vectorize-hulu-batch-and-metrics (batched BatchProcessor_hulu dataset access + fast-path
# support for probabilities/log_probabilities), and reduced to the 3 best-performing metrics
# (no_norm, renyi_divergence alpha=4, renyi_divergence alpha=0.25) to actually engage the fast
# path (the other 6 original metrics force the slow per-token loop regardless of the perf fix).
# Written to noise_tuning_perf/ (NOT noise_tuning/) so run_0's original 6 completed gn_sets
# (full 9-metric config) are untouched, not overwritten.
# Reuses run_0's own already-generated descriptions (model/prompt/image-dependent only, doesn't
# change with the metric-set reduction) for every gn_set, skipping regeneration entirely.

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
out_dir="${run_dir}/noise_tuning_perf"
descriptions_path="${run_dir}/noise_tuning/gn_set0/datasets/generated_descriptions.json"

for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nENDOSCOPY RUN0 PERF NOISE TUNING -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n"

  python /local/scratch/clo37/tmp_figure3_repro_compare/hulu_perf_worktree/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Noise-level tuning for endoscopy run_0 (perf branch, reduced metrics), std set: ${STD_SETS[$set]}'" \
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
