#!/bin/bash
#SBATCH --job-name=test_run_v1
#SBATCH --output=out_test_run_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv

member_dataset='/local/scratch/clo37/datasets/PubMedVision/subsets'
nonmember_dataset='/local/scratch/clo37/datasets/ShareGPT-4o/global_nonmember_fullset.json'
reference_dataset='/local/scratch/clo37/datasets/ShareGPT-4o/global_nonmember_fullset.json'
model='hulu_med'
out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/test_results'
python_path='/local/scratch/clo37/med_vlm_mia/'
NUM_OF_RUNS=1


STD_SETS=(
  "[0.005,0.0078,0.012,0.019,0.03,0.046,0.072,0.11]"
  "[0.18,0.28,0.43,0.67,1.1,1.6,2.6,4.0]"
  "[6.2,9.8,15,24,37,58,91,140]"
  "[220,340,540,840,1300,2100,3200,5000]"
)


export PYTHONPATH=$PYTHONPATH:${python_path}
for ((run=0; run<NUM_OF_RUNS; run++)); do
  for ((set=0; set<1; set++)); do
    printf "\n>>>===================\n\nUsing STD set $set: ${STD_SETS[$set]}\n\n=================== \n\n"
    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Noise-level tuning for dataset: '" \
        job_meta_params.job_type=hyperparam_tuning \
        \
        path.output_dir=${out_dir}/run_${run}/gn_set${set} \
        \
        target_model=${model} \
        \
        data.save_datasets=true \
        data.target_set_size= \
        data.n_nm_ratio=0.5 \
        data.member_dataset=${member_dataset} \
        data.nonmember_dataset=${nonmember_dataset} \
        data.reference_datasets_list=${reference_dataset} \
        data.reference_set_sample_distribution=[] \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_1_kl_div","max_k_renyi_2_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_025","max_k_renyi_divergence_05","max_k_renyi_divergence_2","max_k_renyi_divergence_4"] \
        img_metrics.get_raw_meta_metrics=['losses'] \
        img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_1_kl_div_tkn_vals','max_k_renyi_2_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_025_tkn_vals','max_k_renyi_divergence_05_tkn_vals','max_k_renyi_divergence_2_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
        \
        img_metrics.get_proc_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=0 \
        img_metrics.get_raw_meta_examples=1000 \
        \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]' \
        data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomRotation.use=false
  done
done
