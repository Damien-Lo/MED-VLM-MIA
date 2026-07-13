#!/bin/bash
#SBATCH --job-name=7B_mia_run
#SBATCH --output=out_7B_mia_run.log
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
  printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} \n\n=================== \n\n"
  out_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2026_04_14_test_different_loss/7B/run_${run}
  target_dataset=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2026_04_14_test_different_loss/7B/datasets/target_dataset.parquet
  for ((set=0; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nRUNING FOR MODALILTY: STD $set: ${STD_SETS[$set]} \n\n=================== \n\n"
    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'LOGAN for flickr and coco joined data that should exist on different parts of the loss landscape with llava 7B set: ${STD_SETS[$set]}'" \
        job_meta_params.job_type=evaluation \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model="llava-v1.5-7b" \
        target_model.model_path='liuhaotian/llava-v1.5-7b' \
        \
        data.save_datasets=false \
        data.target_set_size=300 \
        data.n_nm_ratio=0.5 \
        data.dataset=${target_dataset} \
        data.pre_gen_descriptions=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/LatestResults/2026_04_14_test_different_loss/7B/datasets/generated_descriptions.json \
        data.reference_datasets_list="" \
        data.reference_set_sample_distribution="[]" \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_4"] \
        img_metrics.get_raw_meta_metrics=["per_token_CE_loss"] \
        img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
        \
        img_metrics.get_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=2 \
        \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
        data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
        data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
        data.augmentations.RandomRotation.use=false \
        data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0]' \
        data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false
  done
done