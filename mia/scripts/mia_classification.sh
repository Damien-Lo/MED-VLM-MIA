#!/bin/bash


# Load environment
source ~/.bashrc
conda activate 

target_dataset=''
model=''
out_dir=''


OPTIMAL_STD='[]'


export PYTHONPATH=$PYTHONPATH:${python_path}

for ((set=0; set<1; set++)); do
    printf "\n>>>===================\n\nUsing STD set $set: ${STD_SETS[$set]}\n\n=================== \n\n"
    python mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Evaliation at optimal noise for target dataset: '" \
        job_meta_params.job_type=Evaluation \
        \
        path.output_dir=${out_dir}/run_${run}/gn_set${set} \
        \
        target_model=${model} \
        \
        data.dataset=${target_dataset}\
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
        data.augmentations.GaussianNoise.mean='[0.0]' \
        data.augmentations.GaussianNoise.std=${OPTIMAL_STD} \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomRotation.use=false
done

