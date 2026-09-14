#!/bin/bash
#SBATCH --job-name=tune_and_eval
#SBATCH --output=out_tune_and_eval.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week


# job_type=tune_and_eval: runs the hyperparameter-tuning sweep (target set vs. reference set,
# across the full noise range) and the real MIA eval (target set members vs. non-members,
# same as job_type=evaluation) in a single inference pass per std, instead of two separate
# mia.py runs -- see mia.py's job_type=="tune_and_eval" branch / src/eval/eval.py's
# compute_tuning_separation + slice_preds.
#
# Pass a *frozen* target_dataset.parquet and reference_dataset.parquet below (both target_set
# and reference_set need to be pre-built and controlled -- see build_dataset.sh, which now
# also freezes reference_dataset.parquet alongside target_dataset.parquet whenever
# data.reference_datasets_list is given, via build_and_save_reference_set).
#
# Outputs per gn_set${set} dir:
#   Tuning (target-set vs. reference-set, full noise range):
#     tuning_raw_scores.json  -- raw per-metric/per-setting scores split into {"target":[...], "reference":[...]}, for plotting divergence-vs-noise curves
#     tuning_separation.json  -- {"separation": <mean_target/mean_reference/gap/tune_auc per setting>, "best_by_setting": <largest-separation std per metric/k, by both raw gap and tune_auc>}
#   Eval (target-set only, real member vs. non-member -- identical shape to job_type=evaluation):
#     preds.json, auc.json, acc.json, tpr_low_fpr.json, class_labels.json, sampled_raw_meta.json, proc_meta.json

# See scripts/hard_BBC/mia_run_32B_mri.sh for the full list of valid img_metrics.metrics_to_use
# / img_metrics.get_proc_meta_metrics entries.


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


# --- Pass your pre-built target set and reference set here ---
target_dataset='/local/scratch/clo37/MED-VLM-MIA-DATA/results/CHANGE_ME/datasets/target_dataset.parquet'
reference_dataset='/local/scratch/clo37/MED-VLM-MIA-DATA/results/CHANGE_ME/datasets/reference_dataset.parquet'
pre_gen_descriptions='/local/scratch/clo37/MED-VLM-MIA-DATA/results/CHANGE_ME/datasets/generated_descriptions.json'
out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/CHANGE_ME/tune_and_eval'


for ((set=0; set<${#STD_SETS[@]}; set++)); do
  printf "\n>>>===================\n\nRUNING FOR STD $set: ${STD_SETS[$set]} \n\n=================== \n\n"
  python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Combined hyperparameter tuning (target vs reference) and MIA eval (target members vs nonmembers) set: ${STD_SETS[$set]}'" \
      job_meta_params.job_type=tune_and_eval \
      \
      path.output_dir=${out_dir}/gn_set${set} \
      \
      target_model="med_hulu" \
      target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
      \
      data.save_datasets=true \
      data.dataset=${target_dataset} \
      data.reference_dataset=${reference_dataset} \
      data.pre_gen_descriptions=${pre_gen_descriptions} \
      \
      img_metrics.parts=["img"] \
      img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_inf_kl_div"]' \
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
