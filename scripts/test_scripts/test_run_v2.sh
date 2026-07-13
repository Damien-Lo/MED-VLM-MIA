#!/bin/bash
#SBATCH --job-name=14B_test_run_v2
#SBATCH --output=out_test_run_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G


# Load environment
source ~/.bashrc
conda activate med_vlm_mia_venv

target_dataset='/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_02_05/datasets/target_dataset.parquet'

export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
    job_meta_params.test_run=false \
    job_meta_params.description="'Test Run 2 seeing if inference can be made with smaller models'" \
    job_meta_params.job_type=evaluation \
    \
    path.output_dir=/home/clo37/priv/MED-VLM-MIA/test_results/TEST_MIA2 \
    \
    target_model="med_hulu" \
    target_model.model_path='ZJU-AI4H/Hulu-Med-7B' \
    \
    data.save_datasets=true \
    data.target_set_size=300 \
    data.n_nm_ratio=0.5 \
    data.dataset=${target_dataset} \
    data.pre_gen_descriptions='/home/clo37/priv/MED-VLM-MIA/test_results/TEST_MIA1/datasets/generated_descriptions.json' \
    data.reference_datasets_list=/home/clo37/priv/MED-VLM-MIA/test_datasets/test1/reference_dataset.parquet \
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
    

        # data.pre_gen_descriptions='/local/scratch/clo37/MED-VLM-MIA-DATA/test_results/TEST_MIA1/datasets/generated_sentences.json' \