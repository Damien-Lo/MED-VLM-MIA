#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=out_build_dataset.log
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv



export PYTHONPATH=$PYTHONPATH:${python_path}

member_dataset='/local/scratch/clo37/datasets/ROCOv2-radiology/meta_radiology_rocov2.json'
nonmember_dataset='/local/scratch/clo37/datasets/ROCOv2-radiology/synthetic_non_members/gen_radiology_roco_distribution_lim1_1000_exs.json'
reference_dataset='/local/scratch/clo37/datasets/ROCOv2-radiology/synthetic_non_members/gen_radiology_roco_distribution_lim1_1000_exs.json'
python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="Building dataset synthetic generated radiology images from rocov2 with rocov2 as members still somewhat fails binary classifer test but better than before" \
    job_meta_params.job_type='build_dataset' \
    \
    path.output_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_04_25_synth_non-members/mri/run_0 \
    \
    target_model="med_hulu" \
    target_model.model_path='ZJU-AI4H/Hulu-Med-14B' \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.member_dataset=${member_dataset} \
    data.nonmember_dataset=${nonmember_dataset} \
    data.reference_datasets_list=${reference_dataset} \
    data.reference_set_sample_distribution=[] \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=[] \
    img_metrics.get_raw_meta_metrics=['losses'] \
    img_metrics.get_proc_meta_metrics=[]
