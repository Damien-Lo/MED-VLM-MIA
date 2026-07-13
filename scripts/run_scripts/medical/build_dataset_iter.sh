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

# member_dataset='/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_limit_1_images/microscopy_lim1_132978_exs.json'
# nonmember_dataset='/local/scratch/clo37/datasets/Broad_Bioimage_Benchmark_Collection/Kaggle_2018_Data_Science_Bowl/meta_microscopy_full.json'
# reference_dataset='/local/scratch/clo37/datasets/Broad_Bioimage_Benchmark_Collection/Kaggle_2018_Data_Science_Bowl/meta_microscopy_full.json'
# model='med_hulu'
# out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_02_18/14B_exact_one_image/endoscopy/run_1'
# python_path='/local/scratch/clo37/med_vlm_mia/'



modalities=(
    endoscopy
    microscopy
    mri
    endoscopy
    microscopy
    mri
)


# modalities_files=(
#     computed_tomography_lim3_185326_exs.json
#     digital_photography_lim3_48203_exs.json
#     endoscopy_lim3_14483_exs.json
#     fundus_photography_lim3_2867_exs.json
#     infrared_imaging_lim3_3256_exs.json
#     magnetic_resonance_imaging_lim3_106196_exs.json
#     microscopy_lim3_165878_exs.json
#     optical_coherence_tomography_lim3_5509_exs.json
#     other_lim3_50536_exs.json
#     ultrasound_lim3_26961_exs.json
# )

member_datasets=(
    '/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_with_3_images/endoscopy_with_3_images_890_exs.json'
    '/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_with_3_images/microscopy_with_3_images_11959_exs.json'
    '/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_with_3_images/magnetic_resonance_imaging_with_3_images_7535_exs.json'
    '/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_limit_1_images/endoscopy_lim1_11791_exs.json'
    '/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_limit_1_images/microscopy_lim1_132978_exs.json'
    '/local/scratch/clo37/datasets/PubMedVision/subsets/sorted_modalities_limit_1_images/magnetic_resonance_imaging_lim1_84649_exs.json'

)
nonmember_datasets=(
    '/local/scratch/clo37/datasets/EndoBench/meta_endoscopy_full_stacked_3.json'
    '/local/scratch/clo37/datasets/Broad_Bioimage_Benchmark_Collection/Kaggle_2018_Data_Science_Bowl/meta_microscopy_full_stacked_3.json'
    '/local/scratch/clo37/datasets/TCIA/combined_datasets/05_brain_mri_05_other_mri_stacked_3.json'
    '/local/scratch/clo37/datasets/EndoBench/meta_endoscopy_full.json'
    '/local/scratch/clo37/datasets/Broad_Bioimage_Benchmark_Collection/Kaggle_2018_Data_Science_Bowl/meta_microscopy_full.json'
    '/local/scratch/clo37/datasets/TCIA/combined_datasets/05_brain_mri_05_other_mri.json'
)

out_dirs=(
    '/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_03_29/7B_three_images/endoscopy'
    '/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_03_29/7B_three_images/microscopy'
    '/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_03_29/7B_three_images/mri'
    '/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_03_29/14B_exact_one_image/endoscopy'
    '/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_03_29/14B_exact_one_image/microscopy'
    '/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_03_29/14B_exact_one_image/mri'
)



export PYTHONPATH=$PYTHONPATH:${python_path}

# for ((mod=0; mod<${#modalities[@]}; mod++)); do
for ((set=0; set<${#member_datasets[@]}; set++)); do
    member_dataset=${member_datasets[$set]}
    nonmember_dataset=${nonmember_datasets[$set]}
    reference_dataset=${nonmember_datasets[$set]}
    if ((set < 3)); then
        model='ZJU-AI4H/Hulu-Med-7B'
        description_mod="Building dataset for ${member_dataset} data using ${nonmember_dataset} as nonmembers for 7B with exactly 3 image per sample"
    else
        model='ZJU-AI4H/Hulu-Med-14B'
        description_mod="Building dataset for ${member_dataset} data using ${nonmember_dataset} as nonmembers for 14B with exactly 1 image per sample"   
    fi
    for ((run=0; run<3; run++)); do
        out_dir=${out_dirs[$set]}/run_${run}
        python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
            job_meta_params.test_run=false \
            job_meta_params.description="${description_mod}" \
            job_meta_params.job_type='build_dataset' \
            \
            path.output_dir=${out_dir} \
            \
            target_model="med_hulu" \
            target_model.model_path=${model} \
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
done