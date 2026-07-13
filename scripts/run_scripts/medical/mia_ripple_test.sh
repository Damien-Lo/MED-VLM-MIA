#!/bin/bash
#SBATCH --job-name=mia_ripple_test
#SBATCH --output=out_mia_ripple_test.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G


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
    magnetic_resonance_imaging
)


modalities_files=(
    magnetic_resonance_imaging_lim1_106196_exs.json
)




python_path='/local/scratch/clo37/med_vlm_mia/'


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

for ((run=0; run<1; run++)); do
  for ((mod=0; mod<${#modalities[@]}; mod++)); do
    out_dir=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_02_24_ripple_test/14B_exact_one_image/run_0
    target_dataset=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_02_24_ripple_test/14B_exact_one_image/run_0/datasets/target_dataset.parquet
    for ((set=0; set<${#STD_SETS[@]}; set++)); do
      printf "\n>>>===================\n\nRUNING FOR MODALILTY: ${modalities[$mod]} RUN ${run} STD $set: ${STD_SETS[$set]} \n\n=================== \n\n"
      python /home/clo37/priv/MED-VLM-MIA/mia/mia.py \
          job_meta_params.test_run=false \
          job_meta_params.description="'RUN MIA by MRI modality on 14B Hulumed model with exactly 1 images per sample (same dataset as 2026_02_17 14B test): with MRI member and nonmember data RUN ${run} STD $set: ${STD_SETS[$set]}, testing whether I can induce a perterbation ripple at the start, could on the middle tokens to stablize for members and only look at the end tokens, will also run the standard not cut off divergence metrics to look at the whole sequence length to see if I can spot a ripple or which tokens the ripple coes from'" \
          job_meta_params.job_type=evaluation \
          \
          path.output_dir=${out_dir}/gn_set${set} \
          \
          target_model="med_hulu" \
          target_model.model_path='ZJU-AI4H/Hulu-Med-14B' \
          \
          data.save_datasets=false \
          data.target_set_size=300 \
          data.n_nm_ratio=0.5 \
          data.dataset=${target_dataset} \
          data.pre_gen_descriptions=/local/scratch/clo37/MED-VLM-MIA-DATA/results/2026_02_24_ripple_test/14B_exact_one_image/run_0/baselines/datasets/generated_descriptions.json \
          data.reference_datasets_list="" \
          data.reference_set_sample_distribution="[]" \
          \
          img_metrics.parts='["img"]' \
          img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_divergence_4","max_k_no_norn_kl_div_ripple","max_k_renyi_05_kl_div_ripple","max_k_renyi_divergence_4_ripple"]' \
          img_metrics.get_raw_meta_metrics='["losses"]' \
          img_metrics.get_proc_meta_metrics='["max_k_no_norn_kl_div_tkn_vals","max_k_renyi_05_kl_div_tkn_vals","max_k_renyi_divergence_4_tkn_vals","max_k_no_norn_kl_div_tkn_vals_ripple","max_k_renyi_05_kl_div_tkn_vals_ripple","max_k_renyi_divergence_4_tkn_vals_ripple"]' \
          \
          img_metrics.get_meta_examples=1000 \
          img_metrics.get_token_labels=1000 \
          img_metrics.get_raw_images=5 \
          \
          data.augmentations.RandomResize.use=false \
          data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
          data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
          data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
          data.augmentations.RandomRotation.use=false \
          data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
          data.augmentations.GaussianNoise.use=false \
          data.augmentations.GaussianNoise.mean='[0.0]' \
          data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
          data.augmentations.GaussianNoiseRipple.use=true \
          data.augmentations.GaussianNoiseRipple.mean='[0.0]' \
          data.augmentations.GaussianNoiseRipple.std=${STD_SETS[$set]} \
          data.augmentations.GaussianNoiseRipple.start_patch='[0]' \
          data.augmentations.RandomAffine.use=false \
          data.augmentations.ColorJitter.use=false
    done
  done
done


# img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_05_kl_div","max_k_renyi_inf_kl_div","max_k_renyi_divergence_4","max_k_no_norn_kl_div_ripple","max_k_renyi_05_kl_div_ripple","max_k_renyi_inf_kl_div_ripple","max_k_renyi_divergence_4_ripple"]' \
# img_metrics.get_raw_meta_metrics='["losses"]' \
# img_metrics.get_proc_meta_metrics='["max_k_no_norn_kl_div_tkn_vals","max_k_renyi_05_kl_div_tkn_vals","max_k_renyi_inf_kl_div_tkn_vals","max_k_renyi_divergence_4_tkn_vals","max_k_no_norn_kl_div_tkn_vals_ripple","max_k_renyi_05_kl_div_tkn_vals_ripple","max_k_renyi_inf_kl_div_tkn_vals_ripple","max_k_renyi_divergence_4_tkn_vals_ripple"]' \