# LOGAN: Low Gaussian Noise Membership Inference Attack on Vision–Language Models

LOGAN (Low Gaussian Noise Attack) is a simple yet effective **membership inference attack (MIA)** for vision–language models (VLMs).  
The core idea is to exploit the model’s sensitivity to **low-level Gaussian noise**:

> Under small perturbations, **member samples** tend to be more resilient and exhibit consistently **smaller per-token logit divergence** between perturbed and original inputs than **non-members**.

This repository contains the code to:
- Tune the optimal noise level for a given target model and dataset.
- Run LOGAN as a membership inference classifier.
- Inspect and export raw / processed metrics, including divergence curves and perturbed images.

---

## Assets

- **Paper (LOGAN)**: _[link to be added]_  
- **Project name**: `LOGAN-VLM-MIA` (this repo)

---

## Models

We evaluate LOGAN on two pre-trained vision–language models:

- **LLaVA-v1.5-7B**  
  - Official implementation: see the LLaVA repository.  
  - In this repo, the config file  
    `config/target_model/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5.yaml`  
    should have:
    ```yaml
    model_path: /path/to/downloaded/llava/checkpoint
    ```

- **MiniGPT-4 / MiniGPT-4o**  
  - Official implementation: see the MiniGPT-4 repository.  
  - In this repo, the config file  
    `config/target_model/minigpt-4.yaml`  
    should have:
    ```yaml
    ckpt: /path/to/downloaded/minigpt-4o/checkpoint
    ```

You will need to download checkpoints following each model’s official instructions and update the paths in the config files accordingly.

---

## Evaluated Datasets

We evaluate LOGAN on three datasets:

1. **Flickr pre-training data**  
   - Used for: **LLaVA**, **MiniGPT-4o**  
   - Source: [`JaineLi/VL-MIA-image`](https://huggingface.co/datasets/JaineLi/VL-MIA-image)  
   - You should download a local copy and split it into **member** and **non-member** JSON files with format:
     ```json
     [
       {
         "image": "<image_path>",
         "label": "<label>"
       },
       {
         "image": "<image_path>",
         "label": "<label>"
       }
     ]
     ```

2. **COCO-2017 instruction fine-tuning data**  
   - Used for: **LLaVA**  
   - Source: [`liuhaotian/LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)  
   - Store a JSON with format:
     ```json
     [
       {
         "image": "<image_path>",
         "label": 1
       },
       {
         "image": "<image_path>",
         "label": 1
       }
     ]
     ```

3. **ShareGPT-4o (reference non-member set)**  
   - Used as a **known non-member reference** dataset.  
   - Source: [`OpenGVLab/ShareGPT-4o`](https://huggingface.co/datasets/OpenGVLab/ShareGPT-4o/tree/main)  
   - Store as a JSON with format:
     ```json
     [
       {
         "image": "<image_path>",
         "label": 0
       },
       {
         "image": "<image_path>",
         "label": 0
       }
     ]
     ```

---

## Contents

- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Noise-level Tuning](#noise-level-tuning)
- [MIA Classification (Evaluation)](#mia-classification-evaluation)
- [Additional Optional Run Parameters](#additional-optional-run-parameters)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Setup

1. **Clone this repository**
   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>


2. **Install the environment**
   ```bash
   cd scripts
   sbatch install_enviornment.sh
   
This script should install all required packages (e.g., via conda / pip) for running LOGAN.


3. **Download and Install Target Models**
-   Download a LLaVA checkpoint from:
    https://github.com/haotian-liu/LLaVA\
-   Download a MiniGPT-4 checkpoint from the official repository:
    https://github.com/Vision-CAIR/MiniGPT-4

4. **Define the Required Paths**

    Edit the following files: - `config/path/path.yaml`: specify
    `cache_dir` -
    `config/target_model/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5.yaml`:
    set `model_path` - `config/target_model/minigpt-4.yaml`: set `ckpt` to
    MiniGPT-4 checkpoint


## Noise-Level Tuning

You will run `noise_level_tuning.sh` which loops over all standard
deviation sets given in `STD_SETS`.

-   Test 32 noise levels (4 batches × 8 SD values)
-   Choose model: `"llava-v1.5-7b"` or `"minigpt-4"`
-   `out_dir` stores raw divergence arrays, loss, and metadata

### Target Set Creation
The target sets are loaded given a the specified member.dataset and nonmember.dataset, and behave differently dependeing in the file type provided:  
-   If JSON files are provided, a new target set is sampled using\
    `data.n_nm_ratio` and `data.target_set_size`, then saved as Parquet.
-   If Parquet files are provided, the target set is simply loaded from the Parquet files
Therefore, when running a for a new target set, provide JSON files to be sampled from. When continuing tuning on different standard devations of the same target set, provide Parquet files.

### Reference Set Options
The reference sets are additioanlyl loaded differently depending on the file type and parameters provided:
- For the weakest assumption setting where the reference set is exactly the same samples as the target non-members, specify reference_dataset as an empty list: '[]'
- For a slightly stronger assumption where the reference set is sampled from the same parent set distribution as the target non-members, specify reference_dataset as the string path to the parent dataset: "`non_member_parent_set_path`"
- For a slightly stronger assuption where the reference set is sampled from multiple parent datasets, specify the reference_dataset as a list of string paths to the parent sets: "[`path1`, `path2`]". Additionally specify the proportion taken from each set from data.reference_set_sample_distribution: [0.5, 0.5]. Note these proportions have to sum to 1.0 or an error will be raised.
- For a slightly stronger assumption where the reference set is sampled from the a different distribution as the target non-members, specify reference_dataset as the string path to the parent dataset: "`different_distribution_reference_dataset`"


### Optimal Noise-Level Selection
During noise-level tuning, the raw token-wise divergences at each noise level are saved in the specified out_dir. Using these values, we compute and plot the difference between the average divergence of the target set and the reference set across all tested noise levels, and then identify the minimum of this curve. The noise standard deviation at which this minimum occurs is selected as the optimal standard deviation for the MIA according to our noise-tuning procedure.

## Evaluation

Run:

    mia_classification.py

Set: 
- `OPTIMAL_STD`:  to the previously obtained optimal standard deviation of noise. (This can additionally be set to a list of noise levels like in noise-level tuning to observe the behavior of the target set against multiple noise levels)
- `model`: target model to attack
- `out_dir`: output directory for metric evaluations and additional request files (AUC, ACC, divergence_metrics, raw_images)



## Optional Parameters

Several additional flags control how the run is executed:

- `img_metrics.metrics_to_use`  
  Specifies which evaluation metrics / normalization methods to compute the MIA on.  
  The full list of available metrics is defined in:
  - `config/img_metrics/img_metrics.yaml`

- `img_metrics.get_raw_meta_metrics`  
  Specifies which **raw model outputs** to save (e.g., loss values, raw probability vectors).

- `img_metrics.get_proc_meta_metrics`  
  Specifies which **processed metric values** to save (e.g., token-wise divergences used for noise-level tuning).  
  **Important:** If a metric is used in `img_metrics.metrics_to_use`, the corresponding processed metric **must** be included in `img_metrics.get_proc_meta_metrics`.

- `img_metrics.get_raw_meta_examples`  
  Controls how many **raw meta** examples are saved.

- `img_metrics.get_proc_meta_examples`  
  Controls how many **processed meta** examples are saved.

- `img_metrics.get_raw_images`  
  Specifies how many **raw images** to save across all tested noise levels.  
  If this is set to `x`, the code will save perturbed images for the first `x` member samples and the first `x` non-member samples.


