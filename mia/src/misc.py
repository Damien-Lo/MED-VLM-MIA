import os
import json
import torch
from pathlib import Path
from omegaconf import OmegaConf

def save_to_json(dict_obj, filename, cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.json")
    with open(save_path, "w") as f:
        json.dump(dict_obj, f, indent=4)
    print(f"Saved {filename} to {save_path}")
    
def save_to_pt(tensor, filename, cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.pt")
    torch.save(tensor, save_path)
    print(f"Saved {filename} to {save_path}")


def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode

def save_run_meta(cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"run_parameters.txt")

    # OmegaConf.select with a default, not direct attribute access, for the img_metrics fields:
    # this function is now called for job_type=build_dataset runs too (previously it wasn't
    # called at all there), and several of these fields are declared mandatory (`???`, no
    # default) in config, since only hyperparam_tuning/evaluation/tune_and_eval callers used to
    # need them -- direct access would raise omegaconf.errors.MissingMandatoryValue for a
    # build_dataset invocation that never set them, which is the normal case.
    def _get(path):
        return OmegaConf.select(cfg, path, default="N/A (not set for this job_type)")

    txt = f"""\
        Description: {cfg.job_meta_params.description}
        Seed: {cfg.job_meta_params.seed}
        TEST CASE: {cfg.job_meta_params.test_run}, Only Running on first {cfg.inference.test_number_of_batches} batches
        Run Job Type: {cfg.job_meta_params.job_type}
        Target Model: {cfg.target_model.type}
        Target Member Dataset: {cfg.data.member_dataset}
        Target Non-member Dataset: {cfg.data.nonmember_dataset}
        Reference Datasets Used: {cfg.data.reference_datasets_list}
        Reference Dataset Distribution: {cfg.data.reference_set_sample_distribution}
        Augmentations Used: {cfg.data.augmentations}
        Parts Tested: {_get('img_metrics.parts')}
        Metrics Tested: {_get('img_metrics.metrics_to_use')}

        Requested token labels of first {cfg.img_metrics.get_token_labels} of each class
        Requested Raw Augmented Images of first {cfg.img_metrics.get_raw_images} of each class
        print(f"Requested raw metrics values: {_get('img_metrics.get_raw_meta_metrics')}")
        print(f"Requested process metrics values: {_get('img_metrics.get_raw_meta_metrics')}")
        print(f"Will get {cfg.img_metrics.get_meta_examples} or maximum of member and nonmember lengths")
        """
    Path(save_path).write_text(txt, encoding="utf-8")
    return save_path

def build_descriptions_dataset(cfg):
    
    descriptions = list()
    
    if cfg.data.member_dataset != "":
        member_desc_path = cfg.data.member_desc_path
        with open(member_desc_path, 'r') as f:
            mem_data = json.load(f)
        member_idxs = mem_data['idxs']
        
    if cfg.data.nonmember_dataset != "":
        nonmember_desc_path = cfg.data.nonmember_desc_path
        with open(nonmember_desc_path, 'r') as f:
            nonmem_data = json.load(f)
        nonmember_idxs = nonmem_data['idxs']
    
    if cfg.data.nonmember_dataset != "" and cfg.data.member_dataset != "": 
        descriptions.extend(mem_data["sentences"])
        descriptions.extend(nonmem_data['sentences'])
        return member_idxs, nonmember_idxs, descriptions
    
    elif cfg.data.dataset != "":
        print("Loading Single Description")
        with open(cfg.data.single_desc_path, 'r') as f:
            desc = json.load(f)
        return [],[], desc['sentences']
    else:
        raise ValueError(f"No Descriptions Passed")
        