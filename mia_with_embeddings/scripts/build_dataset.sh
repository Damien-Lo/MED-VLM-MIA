#!/bin/bash
#SBATCH --job-name=finetuned_build_dataset
#SBATCH --output=out_finetuned_build_dataset.log
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --partition=h200hour


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv


member_sizes=150
epochs=(1 3 5 7 9)
epochs=(9 5 1)

export PYTHONPATH=$PYTHONPATH:${python_path}

for epoch in "${epochs[@]}"; do
    nonmember_size=$((5407 - member_size))

    echo "Processing member size: $member_size, non-member size: $nonmember_size"

    member_dataset="/local/scratch/clo37/datasets/TCIA/MIDI-B-Curated-Test-1000/additional_hulu_FT_sets/different_member_set_sizes/member_samples_150/member_set_150.json"
    nonmember_dataset="/local/scratch/clo37/datasets/TCIA/MIDI-B-Curated-Test-1000/additional_hulu_FT_sets/different_member_set_sizes/member_samples_150/non_member_set_5257.json"
    reference_dataset="/local/scratch/clo37/datasets/TCIA/MIDI-B-Curated-Test-1000/additional_hulu_FT_sets/different_member_set_sizes/member_samples_150/non_member_set_5257.json"
    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="Building dataset for additionally fintuned LORA adapters with ${member_size} members and ${nonmember_size} non-members" \
        job_meta_params.job_type='build_dataset' \
        \
        path.output_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_20_testing_vision_embeddings/epoch_${epoch}/run_0 \
        \
        target_model="med_hulu" \
        target_model.model_path='ZJU-AI4H/Hulu-Med-7B' \
        target_model.adapter_path="/local/scratch/clo37/models/Hulu-Med-7B/finetuning/member_size_stagered_checkpoints/member_samples_${member_size}" \
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

done
