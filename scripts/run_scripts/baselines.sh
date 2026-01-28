#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=out_baselines.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv

target_dataset='/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/target_dataset.parquet'


STD_SETS=(
  "[0.005]" "[0.0078]"
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


out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_01_25/baselines'
python_path='/local/scratch/clo37/med_vlm_mia/'


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/
python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Baselines for first full run of hulu'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=${out_dir}\
    \
    target_model="med_hulu" \
    target_model.model_path='ZJU-AI4H/Hulu-Med-32B' \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions=/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/generated_sentences.json \
    data.reference_datasets_list=/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/reference_dataset.parquet \
    data.reference_set_sample_distribution=[] \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=["mink","mod_renyi_1_entro","mod_renyi_05_entro","mod_renyi_2_entro","mod_renyi_05_entro","mod_renyi_2_entro","cross_entropy_mink","cross_entropy_diff_mink","per_token_CE_loss","aug_kl"] \
    img_metrics.get_raw_meta_metrics=[] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.get_proc_meta_examples=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=0 \
    img_metrics.get_raw_meta_examples=1000 \
    \
    # data.pre_gen_descriptions='/local/scratch/clo37/MED-VLM-MIA-DATA/test_results/TEST_MIA1/datasets/generated_sentences.json' \