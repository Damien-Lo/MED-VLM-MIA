#!/bin/bash
#SBATCH --job-name=baselines_7B
#SBATCH --output=out_baselines_7B.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G


# Load environment
source ~/.bashrc
conda activate vlm_mia_llava_minigpt_venv


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


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/



for ((run=0; run<1; run++)); do
printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} RUN: ${run} \n\n=================== \n\n"
out_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2026_04_14_test_different_loss/7B/baselines
target_dataset=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2026_04_14_test_different_loss/7B/datasets/target_dataset.parquet
python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Baselines for flickr and coco joined data that should exist on different parts of the loss landscape with llava 7B'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=${out_dir} \
    \
    target_model="llava-v1.5-7b" \
    target_model.model_path='liuhaotian/llava-v1.5-7b' \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions="" \
    \
    img_metrics.parts=["img"] \
    img_metrics.metrics_to_use=['aug_kl','max_k_renyi_1_entro','max_k_renyi_05_entro','mink'] \
    img_metrics.get_raw_meta_metrics=['per_token_CE_loss'] \
    img_metrics.get_proc_meta_metrics=[] \
    \
    img_metrics.get_meta_examples=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=5 \
    \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomRotation.use=true \
    data.augmentations.GaussianNoise.use=false \
    data.augmentations.RandomAffine.use=true \
    data.augmentations.ColorJitter.use=true
done
