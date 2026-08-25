#!/bin/bash
#SBATCH --job-name=finetuned_build_dataset
#SBATCH --output=out_finetuned_build_dataset.log
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv


export PYTHONPATH=$PYTHONPATH:${python_path}


member_dataset="/local/scratch/clo37/datasets/Binary_Classifer_Filtered_Sets/mri_sets/member_meta_radiology_rocov2_hard.json"
nonmember_dataset="/local/scratch/clo37/datasets/Binary_Classifer_Filtered_Sets/mri_sets/non_members_roco_prompt_gen_radiology_roco_distribution_lim1_1655_exs_hard.json"
reference_dataset="/local/scratch/clo37/datasets/Binary_Classifer_Filtered_Sets/mri_sets/non_members_roco_prompt_gen_radiology_roco_distribution_lim1_1655_exs_hard.json"
python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="Building dataset for PubMedVision members and MRI non-members after BBC filtered" \
    job_meta_params.job_type='build_dataset' \
    \
    path.output_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/mri/run_2 \
    \
    target_model="med_hulu" \
    target_model.model_path='ZJU-AI4H/Hulu-Med-32B' \
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
