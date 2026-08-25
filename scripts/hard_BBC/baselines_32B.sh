#!/bin/bash
#SBATCH --job-name=finetuned_baselines_32B_mri
#SBATCH --output=out_finetuned_baselines_32B_mri.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G


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

# modalities=(
#     computed_tomography
#     digital_photography
#     endoscopy
#     fundus_photography
#     infrared_imaging
#     magnetic_resonance_imaging
#     microscopy
#     optical_coherence_tomography
#     other
#     ultrasound
# )



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


modalities=(
  "mamogram"
  )



export PYTHONPATH=$PYTHONPATH:${python_path}


for ((run=0; run<1; run++)); do
# printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} RUN: ${run} \n\n=================== \n\n"
out_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/mri/run_2/baselines
target_dataset=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_08_12_medical_hard_BBC_score_flip_redo/mri/run_2/datasets/target_dataset.parquet
python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="Baselines for PubMedVision members and mri non-members that pass the hard BBC filter" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=${out_dir} \
    \
    target_model="med_hulu" \
    target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions="" \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=['aug_kl','max_k_renyi_1_entro','max_k_renyi_05_entro','mink'] \
    img_metrics.get_raw_meta_metrics=[] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.get_meta_examples=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=0 \
    \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomRotation.use=true \
    data.augmentations.GaussianNoise.use=false \
    data.augmentations.RandomAffine.use=true \
    data.augmentations.ColorJitter.use=true
done