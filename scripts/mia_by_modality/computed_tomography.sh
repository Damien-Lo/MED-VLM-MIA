#!/bin/bash
#SBATCH --job-name=computed_tomography
#SBATCH --output=out_computed_tomography.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv


member_dataset='/local/scratch/clo37/datasets/PubMedVision/subsets'
nonmember_dataset='/local/scratch/clo37/datasets/ShareGPT-4o/global_nonmember_fullset.json'
reference_dataset='/local/scratch/clo37/datasets/ShareGPT-4o/global_nonmember_fullset.json'
model='med_hulu'
out_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_02_11'
python_path='/local/scratch/clo37/med_vlm_mia/'
NUM_OF_RUNS=3

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for ((run=0; run<NUM_OF_RUNS; run++)); do
  for ((set=0; set<1; set++)); do
    python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
        job_meta_params.test_run=true \
        job_meta_params.description="Run ${run}, set ${set} of full MIA evaluation for computed_tomography" \
        job_meta_params.job_type=evaluation \
        \
        path.output_dir=${out_dir}/run_${run}/gn_set${set} \
        \
        target_model=${model} \
        target_model.model_path='/local/scratch/clo37/models/Hulu-Med-32B' \
        \
        data.save_datasets=true \
        data.target_set_size=300 \
        data.n_nm_ratio=0.5 \
        data.member_dataset=${member_dataset} \
        data.nonmember_dataset=${nonmember_dataset} \
        data.pre_gen_descriptions='' \
        data.reference_datasets_list= \
        data.reference_set_sample_distribution=[] \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div"] \
        img_metrics.get_raw_meta_metrics=['losses'] \
        img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals'] \
        \
        img_metrics.get_meta_examples=1000 \
        img_metrics.get_token_labels=1000 \
        img_metrics.get_raw_images=5 \
        \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomResize.size='[[256,256]]' \
        data.augmentations.RandomResize.scale='[[0.2,0.2]]' \
        data.augmentations.RandomResize.ratio='[[0.75,0.75]]' \
        data.augmentations.RandomRotation.use=false \
        data.augmentations.RandomRotation.degrees='[]' \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0]' \
        data.augmentations.GaussianNoise.std='[2.5]' \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false \
        