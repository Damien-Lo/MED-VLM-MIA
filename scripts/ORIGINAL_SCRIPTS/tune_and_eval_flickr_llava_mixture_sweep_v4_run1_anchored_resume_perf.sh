#!/bin/bash
#SBATCH --job-name=flickr_llava_mixture_sweep_v4_resume_perf
#SBATCH --output=/sunhome/clo37/priv/MED-VLM-MIA/out_flickr_llava_mixture_sweep_v4_resume_perf.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --partition=h200week

# Resume of the v4 run1-anchored mixture sweep (originally job 42340, then 42725, cancelled to
# free a GPU slot for profiling) -- now running on the perf/vectorize-batch-and-metrics branch
# (commit cd21617), which fixes the two real bottlenecks profiled on real production data:
#   _get_augmented_batch:      21.2s -> 2.7s/batch  (batched dataset-slice access, not per-row)
#   get_meta_metrics_by_part:  16.5s -> 1.5s/batch  (skips the per-token loop entirely when only
#                                                     the vectorizable metrics are requested)
# Measured end-to-end: ~44.6s/batch -> ~11.0s/batch (~4x), confirmed on real data via tqdm's own
# per-batch timing, not just the profiling instrumentation's sum. Correctness verified against
# the original per-token-loop implementation in mia/tests/test_meta_metrics_vectorized.py (4
# cases: fast/slow path x uniform/ragged-length batches, all pass).
#
# Runs from an isolated git worktree (perf_branch_worktree), never touching the main checkout
# (which stays on MergeTuning) -- avoids any repeat of the earlier near-miss where switching the
# main tree's branch mid-run could have affected a live sweep's fresh `python mia.py` calls.
#
# Picks up exactly where job 42725 left off (p=000,010,020,030,040,050 already complete,
# untouched):
#   p=060: gn_set0 was mid-write when 42725 was cancelled (no tuning_raw_scores.json/preds.json)
#          -- redone from gn_set0.
#   p=070,080,090,100: untouched -- run in full (gn_set0..32).
#
# 2026-09-14 update (job 42855 -> resubmitted as 42865, batch_size 16->32 after confirming ample
# VRAM headroom): ran p=060 (33/33), p=070 (33/33), p=080 (33/33), p=090 (16/33) overnight with
# NO errors and normal per-batch timing -- but all of it turned out to be silently empty. Root
# cause: leftover, UNCOMMITTED profiling instrumentation in this worktree (from the job-42856
# profiling pass) had accidentally mis-indented the line that populates `all_settings_in_aug` in
# renyi_kl_div_maxk (mia/src/metrics/proposed_metrics.py) to sit inside an `if _PROFILE:` block,
# so it silently never ran in production (MIA_PROFILE_INFER was never set) -- producing valid but
# empty preds.json/auc.json/tuning_raw_scores.json every time, no exception anywhere. Confirmed
# via a debug-instrumented test_run diagnostic (job 43116 broken / 43117 fixed) that traced the
# structure through the whole chain. THE ACTUAL COMMITTED/PUSHED PERF-BRANCH CODE (cd21617) WAS
# NEVER AFFECTED -- `git status` on this worktree after removing the stray debug code shows
# "nothing to commit, working tree clean", i.e. byte-identical to cd21617. This was purely an
# artifact of relaunching the real sweep from a worktree that still had temporary debug edits in
# it. Re-ran the correctness test suite (job 43118) as a final sanity check before resuming.
#
# 2026-09-15: since ALL of p=060/070/080/090 (and by extension p=100) ran on the broken worktree
# state, none of that output is usable -- redoing p=060 through p=100 in full from gn_set0. The
# generated_descriptions.json files under each p's gn_set0 are unaffected (description generation
# happens before the broken augmentation/metric step) and are safely reused via pre_gen_arg below.

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

MIXTURE_PS=(060 070 080 090 100)

WORKTREE=/local/scratch/clo37/tmp_figure3_repro_compare/perf_branch_worktree
iclr_dir='/local/scratch/clo37/MED-VLM-MIA-DATA/results/ICLR_additional_results'
target_dataset="${iclr_dir}/datasets/target_dataset_run1_exact.parquet"
mixture_ref_dir_v4="${iclr_dir}/datasets/mixture_reference_sets_v4_run1_anchored"
out_root="${iclr_dir}/v4_run1_anchored/results"


for pp in "${MIXTURE_PS[@]}"; do
  reference_dataset="${mixture_ref_dir_v4}/reference_dataset_p${pp}.parquet"
  out_dir="${out_root}/mix_p${pp}"
  descriptions_path="${out_dir}/gn_set0/datasets/generated_descriptions.json"

  for ((set=0; set<${#STD_SETS[@]}; set++)); do
    printf "\n>>>===================\n\nMIXTURE p=0.%s (v4 run_1-anchored, perf branch) -- STD set $set: ${STD_SETS[$set]} \n\n=================== \n\n" "${pp}"

    pre_gen_arg=""
    if [ "$set" -ne 0 ]; then
      pre_gen_arg="data.pre_gen_descriptions=${descriptions_path}"
    fi

    python ${WORKTREE}/mia/mia.py \
        job_meta_params.test_run=false \
        job_meta_params.description="'Flickr/LLaVA mixture-sweep rerun (v4, run_1-anchored, perf branch): reference = ${pp}% flickr non-members (run_1-anchored) / $((100 - 10#$pp))% ShareGPT, std set: ${STD_SETS[$set]}'" \
        job_meta_params.job_type=tune_and_eval \
        \
        path.output_dir=${out_dir}/gn_set${set} \
        \
        target_model="llava-v1.5-7b" \
        target_model.model_path='liuhaotian/llava-v1.5-7b' \
        inference.batch_size=32 \
        prompt.text="'Describe this image concisely.'" \
        \
        data.save_datasets=false \
        data.dataset=${target_dataset} \
        data.reference_dataset=${reference_dataset} \
        ${pre_gen_arg} \
        \
        img_metrics.parts=["img"] \
        img_metrics.metrics_to_use='["max_k_no_norn_kl_div","max_k_renyi_inf_kl_div"]' \
        img_metrics.get_raw_meta_metrics=[] \
        img_metrics.get_proc_meta_metrics=[] \
        \
        img_metrics.max_k_no_norn_kl_div.ratio='[1.0]' \
        img_metrics.max_k_renyi_inf_kl_div.ratio='[1.0]' \
        \
        img_metrics.get_meta_examples=1000 \
        img_metrics.get_token_labels=0 \
        img_metrics.get_raw_images=0 \
        \
        data.augmentations.RandomResize.use=false \
        data.augmentations.RandomRotation.use=false \
        data.augmentations.GaussianNoise.use=true \
        data.augmentations.GaussianNoise.mean='[0.0]' \
        data.augmentations.GaussianNoise.std=${STD_SETS[$set]} \
        data.augmentations.RandomAffine.use=false \
        data.augmentations.ColorJitter.use=false
  done
done
