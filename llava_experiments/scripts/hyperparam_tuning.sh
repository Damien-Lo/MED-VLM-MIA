#!/bin/bash
#SBATCH --job-name=coco_v1_hyperparam_tuning
#SBATCH --output=out_v1_hyperparam_tuning.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_mia_latest_venv

member_dataset="/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/coco/hyperparam_tuning/coco_2017_IFT_member_ratio_0.5__exact_flickr_KN/gn_set0/datasets/member_target_dataset.parquet"
nonmember_dataset="/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/FINAL_DATA/FINAL_RESULTS/llava/coco/hyperparam_tuning/coco_2017_IFT_member_ratio_0.5__exact_flickr_KN/gn_set0/datasets/non_member_target_dataset.parquet"


STD_SETS=(
  "[0.005,0.0078,0.012,0.019,0.03,0.046,0.072,0.11]"
  "[0.18,0.28,0.43,0.67,1.1,1.6,2.6,4.0]"
  "[6.2,9.8,15,24,37,58,91,140]"
  "[220,340,540,840,1300,2100,3200,5000]"
)


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for ((set=1; set<4; set++)); do
  printf "\n>>>===================\n\nUsing STD set $set: ${STD_SETS[$set]}\n\n=================== \n\n"
  python /home/clo37/priv/VLM-MIA-Study/mia.py \
      job_meta_params.test_run=false \
      job_meta_params.description="'Run hyperparam tuning on target set of flickr 20 members and 200 flicker nonmembers to test the theory of sampling from the member set'" \
      job_meta_params.job_type=hyperparam_tuning \
      \
      path.output_dir=/local/scratch/clo37/VLM_MIA_STUDY_Archive_Data/results/2026_03_16/gn_set${set} \
      \
      target_model="llava-v1.5-7b" \
      \
      data.save_datasets=true \
      data.target_set_size=300 \
      data.n_nm_ratio=0.5 \
      data.member_dataset=${member_dataset} \
      data.nonmember_dataset=${nonmember_dataset} \
      data.reference_datasets_list=[] \
      data.reference_set_sample_distribution=[] \
      \
      img_metrics.parts=["img"] \
      img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"] \
      img_metrics.get_raw_meta_metrics=['losses'] \
      img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_1_kl_div_tkn_vals','max_k_renyi_2_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_025_tkn_vals','max_k_renyi_divergence_05_tkn_vals','max_k_renyi_divergence_2_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
      \
      img_metrics.get_proc_meta_examples=1000 \
      img_metrics.get_token_labels=1000 \
      img_metrics.get_raw_images=1 \
      img_metrics.get_raw_meta_examples=1000 \
      \
      data.augmentations.RandomResize.use=false \
      data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
      data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
      data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
      data.augmentations.RandomRotation.use=false \
      data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
      data.augmentations.GaussianNoise.use=true \
      data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
      data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
      data.augmentations.RandomAffine.use=false \
      data.augmentations.ColorJitter.use=false
done
