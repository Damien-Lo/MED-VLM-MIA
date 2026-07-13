#!/bin/bash
#SBATCH --job-name=finetuned_test_adapter_baseline
#SBATCH --output=out_finetuned_test_adapter_baseline.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv



export PYTHONPATH=$PYTHONPATH:${python_path}

target_dataset=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_05_25_synth_mia_additional_finetuned/run_0/datasets/target_dataset.parquet
python /home/clo37/priv/MED-VLM-MIA/mia/generate_descriptions.py \
    job_meta_params.test_run=false \
    job_meta_params.description="Generating Descriptions with adapter " \
    job_meta_params.job_type='generation' \
    \
    path.output_dir=/home/clo37/priv/MED-VLM-MIA/scripts/additional_finetuned/baseline \
    \
    target_model="med_hulu" \
    target_model.model_path='ZJU-AI4H/Hulu-Med-7B' \
    \
    data.save_datasets=false \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions="" \
    data.reference_datasets_list="" \
    data.reference_set_sample_distribution="[]" \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=[] \
    img_metrics.get_raw_meta_metrics=['losses'] \
    img_metrics.get_proc_meta_metrics=[]
