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

member_dataset='/local/scratch/clo37/datasets/mix_of_flickr_coco/mixed_member_subset.json'
nonmember_dataset='/local/scratch/clo37/datasets/mix_of_flickr_coco/mixed_nonmember_subset.json'
reference_dataset='/local/scratch/clo37/datasets/mix_of_flickr_coco/mixed_nonmember_subset.json'
python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Building dataset with mixed flickr and coco members and nonmemebrs with eual distribution to test datasets with samples that reside in different parts of the loss landscape, need to test if these actually live in different parts of the loss landscape'" \
    job_meta_params.job_type='build_dataset' \
    \
    path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2026_04_14_test_different_loss/7B \
    \
    target_model="llava-v1.5-7b" \
    target_model.model_path='liuhaotian/llava-v1.5-7b' \
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
