#!/bin/bash
#SBATCH --job-name=mia_run_v2
#SBATCH --output=out_mia_run_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv

target_dataset='/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/target_dataset.parquet'


STD_SETS=(
  "[0.005]" "[0.0078]"
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


out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_01_25'
python_path='/local/scratch/clo37/med_vlm_mia/'


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for ((set=0; set<32; set++)); do
    printf "\n>>>===================\n\nUsing STD set $set: ${STD_SETS[$set]}\n\n=================== \n\n"
    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Get baselines for hulumed model full run of MIA evaluation on hulu_med dataset set ${STD_SETS[$set]}'" \
        job_meta_params.job_type=evaluation \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model="med_hulu" \
        target_model.model_path='ZJU-AI4H/Hulu-Med-32B' \
        \
        data.save_datasets=true \
        data.target_set_size=300 \
        data.n_nm_ratio=0.5 \
        data.dataset=${target_dataset} \
        data.pre_gen_descriptions=/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/generated_sentences.json \
        data.reference_datasets_list=/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/reference_dataset.parquet \
        data.reference_set_sample_distribution=[] \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_renyi_05_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_4"] \
        img_metrics.get_raw_meta_metrics=['losses'] \
        img_metrics.get_proc_meta_metrics=['max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
        \
        img_metrics.get_proc_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=0 \
        img_metrics.get_raw_meta_examples=1000 \
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
        # data.pre_gen_descriptions='/local/scratch/clo37/MED-VLM-MIA-DATA/test_results/TEST_MIA1/datasets/generated_sentences.json' \