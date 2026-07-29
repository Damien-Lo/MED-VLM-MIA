#!/bin/bash
#SBATCH --job-name=entropy_gated_test
#SBATCH --output=out_entropy_gated_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Valid entries for img_metrics.metrics_to_use (src/metrics/img_metrics.py):
#   Baseline:        aug_kl, mink, mod_renyi_1_entro, mod_renyi_05_entro, mod_renyi_2_entro,
#                    max_prob_gap, max_k_renyi_1_entro, max_k_renyi_2_entro, max_k_renyi_05_entro,
#                    min_k_renyi_1_entro, min_k_renyi_2_entro, min_k_renyi_05_entro
#   Cross entropy:   cross_entropy_mink, cross_entropy_diff_mink
#   KL divergence:   max_k_no_norn_kl_div, max_k_renyi_05_kl_div, max_k_renyi_1_kl_div,
#                    max_k_renyi_2_kl_div, max_k_renyi_inf_kl_div
#   Renyi divergence: max_k_renyi_divergence_025, max_k_renyi_divergence_05,
#                    max_k_renyi_divergence_2, max_k_renyi_divergence_4
#   Ripple KL div:   max_k_no_norn_kl_div_ripple, max_k_renyi_05_kl_div_ripple,
#                    max_k_renyi_1_kl_div_ripple, max_k_renyi_2_kl_div_ripple,
#                    max_k_renyi_inf_kl_div_ripple
#   Ripple Renyi div: max_k_renyi_divergence_025_ripple, max_k_renyi_divergence_05_ripple,
#                    max_k_renyi_divergence_2_ripple, max_k_renyi_divergence_4_ripple
#
# This is the embeddings variant of the model (src/metrics/vision_embedding_metrics.py),
# so metrics_to_use also accepts these "vision_"-prefixed entries, computed on the raw
# vision-encoder patch embeddings instead of LLM token probabilities. Each one is emitted
# in two variants automatically ("vision" alone, and "vision_img_concat" concatenated with
# the "img" part's scores -- the latter only if "img" is in img_metrics.parts):
#   Vision KL divergence:   vision_max_k_no_norn_kl_div, vision_max_k_renyi_05_kl_div,
#                    vision_max_k_renyi_1_kl_div, vision_max_k_renyi_2_kl_div,
#                    vision_max_k_renyi_inf_kl_div
#   Vision Renyi divergence: vision_max_k_renyi_divergence_025, vision_max_k_renyi_divergence_05,
#                    vision_max_k_renyi_divergence_2, vision_max_k_renyi_divergence_4
#   Vision ripple KL div:   vision_max_k_no_norn_kl_div_ripple, vision_max_k_renyi_05_kl_div_ripple,
#                    vision_max_k_renyi_1_kl_div_ripple, vision_max_k_renyi_2_kl_div_ripple,
#                    vision_max_k_renyi_inf_kl_div_ripple
#   Vision ripple Renyi div: vision_max_k_renyi_divergence_025_ripple, vision_max_k_renyi_divergence_05_ripple,
#                    vision_max_k_renyi_divergence_2_ripple, vision_max_k_renyi_divergence_4_ripple
#   Vision baseline: vision_max_prob_gap, vision_max_k_renyi_1_entro, vision_max_k_renyi_2_entro,
#                    vision_max_k_renyi_05_entro, vision_min_k_renyi_1_entro,
#                    vision_min_k_renyi_2_entro, vision_min_k_renyi_05_entro
#
# Valid entries for img_metrics.get_proc_meta_metrics (gates in src/metrics/meta_metrics.py;
# each only takes effect if the matching metric above is also in metrics_to_use):
#   max_k_no_norn_kl_div_tkn_vals, max_k_no_norn_kl_div_tkn_vals_ripple,
#   max_k_renyi_05_kl_div_tkn_vals, max_k_renyi_05_kl_div_tkn_vals_ripple,
#   max_k_renyi_2_kl_div_tkn_vals, max_k_renyi_2_kl_div_tkn_vals_ripple,
#   max_k_renyi_inf_kl_div_tkn_vals, max_k_renyi_inf_kl_div_tkn_vals_ripple
# NOTE: there is no vision_ counterpart for get_proc_meta_metrics -- src/inference/loop.py:144
# calls get_vision_embedding_metrics() and discards its per-token meta return entirely
# (`batch_vision_pred, _ = ...`), so vision_*_tkn_vals are never written to proc_meta.json
# regardless of what's listed here.


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


modalities=(
  "mamogram"
  )




python_path='/local/scratch/clo37/med_vlm_mia/'


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

# member_sizes=(150 200 250 500 750 1000 1500 2000 2703)

# member_sizes=(150)

# member_sizes=(200 250 500 750 1000 1500 2000 2703)


epochs=(9 7 5 3 1)

epochs=(9 5 1)


export PYTHONPATH=$PYTHONPATH:${python_path}

for epoch in "${epochs[@]}"; do

    echo "Processing epoch: $epoch"

    for ((run=0; run<1; run++)); do
      # printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} \n\n=================== \n\n"
      out_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_21_entropy_gated_test/epoch_${epoch}/run_${run}
      target_dataset=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_21_entropy_gated_test/epoch_${epoch}/run_${run}/datasets/target_dataset.parquet
      for ((set=0; set<${#STD_SETS[@]}; set++)); do
        # printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} RUN ${run} STD $set: ${STD_SETS[$set]} \n\n=================== \n\n"
        python /home/clo37/priv/MED-VLM-MIA/mia_with_embeddings/mia.py \
            job_meta_params.test_run=false \
            job_meta_params.description="Test entropy-gated KLD metrics for 7B on refinetuned hulumed with TCIA mamograms trained on 150 members with ${epoch} epochs" \
            job_meta_params.job_type=evaluation \
            \
            path.output_dir=${out_dir}/gn_set${set} \
            \
            target_model="med_hulu" \
            target_model.model_path='/local/scratch/clo37/models/Hulu-Med-7B-embeddings' \
            target_model.adapter_path="/local/scratch/clo37/models/Hulu-Med-7B-embeddings/finetuning/various_epochs_150_samples_training/epoch_${epoch}" \
            \
            data.save_datasets=false \
            data.target_set_size=300 \
            data.n_nm_ratio=0.5 \
            data.dataset=${target_dataset} \
            data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_21_entropy_gated_test/epoch_${epoch}/run_${run}/baselines/datasets/generated_descriptions.json \
            data.reference_datasets_list="" \
            data.reference_set_sample_distribution="[]" \
            \
            img_metrics.parts=["img"] \
            img_metrics.metrics_to_use=["entropy_gated_no_norn_kl_div","entropy_gated_renyi_05_kl_div","entropy_gated_renyi_1_kl_div","entropy_gated_renyi_2_kl_div","entropy_gated_renyi_inf_kl_div"] \
            img_metrics.get_raw_meta_metrics=['probabilities'] \
            img_metrics.get_proc_meta_metrics=[] \
            \
            img_metrics.get_meta_examples=0 \
            img_metrics.get_token_labels=1000 \
            img_metrics.get_raw_images=0 \
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
done




# modalities=(
#   "mamogram"
#   )




# python_path='/local/scratch/clo37/med_vlm_mia/'


# export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/


# for ((run=0; run<1; run++)); do
#   for ((mod=0; mod<${#modalities[@]}; mod++)); do
#     printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} \n\n=================== \n\n"
#     out_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_05_25_with_descriptions/run_${run}
#     target_dataset=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_05_25_with_descriptions/run_0/datasets/target_dataset.parquet
#     for ((set=0; set<${#STD_SETS[@]}; set++)); do
#       printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} RUN ${run} STD $set: ${STD_SETS[$set]} \n\n=================== \n\n"
#       python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
#           job_meta_params.test_run=false \
#           job_meta_params.description="RUN MIA for 7B on hulumed extra fine tuned with TCIA mamogram images using part: img_inst_desc" \
#           job_meta_params.job_type=evaluation \
#           \
#           path.output_dir=${out_dir}/gn_set${set} \
#           \
#           target_model="med_hulu" \
#           target_model.model_path='ZJU-AI4H/Hulu-Med-7B' \
#           target_model.adapter_path='/local/scratch/clo37/models/Hulu-Med-7B/finetuning/additional_checkpoints' \
#           \
#           data.save_datasets=false \
#           data.target_set_size=300 \
#           data.n_nm_ratio=0.5 \
#           data.dataset=${target_dataset} \
#           data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_05_25_with_descriptions/run_0/baselines/datasets/generated_descriptions.json \
#           data.reference_datasets_list="" \
#           data.reference_set_sample_distribution="[]" \
#           \
#           img_metrics.parts=["img_inst_desc"] \
#           img_metrics.metrics_to_use=["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_4"] \
#           img_metrics.get_raw_meta_metrics=['losses'] \
#           img_metrics.get_proc_meta_metrics=['max_k_no_norn_kl_div_tkn_vals','max_k_renyi_05_kl_div_tkn_vals','max_k_renyi_inf_kl_div_tkn_vals','max_k_renyi_divergence_4_tkn_vals'] \
#           \
#           img_metrics.get_meta_examples=1000 \
#           img_metrics.get_token_labels=1000 \
#           img_metrics.get_raw_images=2 \
#           \
#           data.augmentations.RandomResize.use=false \
#           data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
#           data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
#           data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
#           data.augmentations.RandomRotation.use=false \
#           data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
#           data.augmentations.GaussianNoise.use=true \
#           data.augmentations.GaussianNoise.mean='[0.0]' \
#           data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
#           data.augmentations.RandomAffine.use=false \
#           data.augmentations.ColorJitter.use=false
#     done
#   done
# done