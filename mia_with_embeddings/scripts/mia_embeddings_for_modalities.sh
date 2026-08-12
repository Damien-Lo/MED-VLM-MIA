#!/bin/bash
#SBATCH --job-name=modalitiy_vision_embeddings_mia_run_7B
#SBATCH --output=out_modalitiy_vision_embeddings_mia_run_7B.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=h200week


# TODO: NOTE: Currently, with no fixed seed for each std run, if sampling (meta or proc) is not 1000, then ALL runs will have different samples, so the results are not directly comparable. If you want to compare across runs, you should set a fixed seed for each run (e.g. 0, 1, 2, etc.) and use that same seed for all runs. This is especially important if you are using get_proc_meta_metrics, since the token-level metrics will be computed on different tokens for each run if the samples are different.

# Valid entries for img_metrics.metrics_to_use (src/metrics/img_metrics.py):
#   Baseline:        aug_kl, mink, mod_renyi_1_entro, mod_renyi_05_entro, mod_renyi_2_entro,
#                    max_prob_gap, max_k_renyi_1_entro, max_k_renyi_2_entro, max_k_renyi_05_entro,
#                    min_k_renyi_1_entro, min_k_renyi_2_entro, min_k_renyi_05_entro
#   Cross entropy:   cross_entropy_mink, cross_entropy_diff_mink
#   KL divergence:   max_k_no_norn_kl_div, max_k_renyi_05_kl_div, max_k_renyi_1_kl_div,
#                    max_k_renyi_2_kl_div, max_k_renyi_inf_kl_div
#   Entropy-gated KL divergence: same per-token KL divergences as above, but instead
#                    of averaging the top-k highest-divergence tokens, averages the
#                    divergence at the k tokens with the LOWEST entropy on the original
#                    (unperturbed) input -- i.e. the model's most confident, "deep member
#                    well" tokens (src/metrics/proposed_metrics.py:entropy_gated_kl_div_mink):
#                    entropy_gated_no_norn_kl_div, entropy_gated_renyi_05_kl_div,
#                    entropy_gated_renyi_1_kl_div, entropy_gated_renyi_2_kl_div,
#                    entropy_gated_renyi_inf_kl_div
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
#   entropy_gated_no_norn_kl_div_tkn_vals, entropy_gated_renyi_05_kl_div_tkn_vals,
#   entropy_gated_renyi_1_kl_div_tkn_vals, entropy_gated_renyi_2_kl_div_tkn_vals,
#   entropy_gated_renyi_inf_kl_div_tkn_vals
# The entropy_gated_* metrics above also always write a *_tkn_order companion array to
# proc_meta.json (entropy_gated_no_norn_kl_div_tkn_order, etc.) -- for each sample, the
# full list of token indices in ascending original-entropy order, so any ratio's min-k
# selection is just a prefix slice of that list (no separate config flag needed for it).
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
  "endoscopy"
  "mri"
  "microscopy"
  )




python_path='/local/scratch/clo37/med_vlm_mia/'


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

# member_sizes=(150 200 250 500 750 1000 1500 2000 2703)

# member_sizes=(150)

# member_sizes=(200 250 500 750 1000 1500 2000 2703)


epochs=(9 7 5 3 1)

epochs=(9 5 1)


export PYTHONPATH=$PYTHONPATH:${python_path}

# for epoch in "${epochs[@]}"; do

#     echo "Processing epoch: $epoch"
for mod in "${modalities[@]}"; do
    echo "Processing modality: $mod"
    for ((run=0; run<1; run++)); do
        # printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} \n\n=================== \n\n"
        out_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_31_confirm_no_change_with_vision_med_modalities/${mod}/run_${run}
        target_dataset=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_31_confirm_no_change_with_vision_med_modalities/${mod}/datasets/target_dataset.parquet
        for ((set=0; set<${#STD_SETS[@]}; set++)); do
        # printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} RUN ${run} STD $set: ${STD_SETS[$set]} \n\n=================== \n\n"
        python /home/clo37/priv/MED-VLM-MIA/mia_with_embeddings/mia.py \
            job_meta_params.test_run=false \
            job_meta_params.description="MIA using vision embeddings for further analysis for roco images that failed BBC to see if there is any improvement" \
            job_meta_params.job_type=evaluation \
            \
            path.output_dir=${out_dir}/gn_set${set} \
            \
            target_model="med_hulu" \
            target_model.model_path='/local/scratch/clo37/models/Hulu-Med-7B-embeddings' \
            \
            data.save_datasets=false \
            data.target_set_size=300 \
            data.n_nm_ratio=0.5 \
            data.dataset=${target_dataset} \
            data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_07_31_confirm_no_change_with_vision_med_modalities/${mod}/baselines/datasets/generated_descriptions.json \
            data.reference_datasets_list="" \
            data.reference_set_sample_distribution="[]" \
            \
            img_metrics.parts=["img"] \
            img_metrics.metrics_to_use=['vision_max_k_renyi_divergence_05','vision_max_k_renyi_inf_kl_div','vision_max_k_renyi_divergence_4'] \
            img_metrics.get_raw_meta_metrics=[] \
            img_metrics.get_proc_meta_metrics=[] \
            \
            img_metrics.get_meta_examples=1000 \
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