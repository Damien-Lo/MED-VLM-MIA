import numpy as np
from collections import defaultdict
import copy
import sys
from torchvision import transforms

"""
Main MIA entry point
"""
import os
import json
import hydra
import torch
from src.eval import evaluate
from src.inference import inference
from src.data import get_mod_infer_data
from src.data import get_generation_data
from src.data.generate_descriptions import generate_descriptions
from src.model import load_target_model
from textwrap import dedent
from src.misc import save_to_json, save_to_pt, save_run_meta, build_descriptions_dataset
from src.data.build_dataset_methods import build_target_set, json_to_dataset, list_to_dataset, get_random_subset, build_reference_set, build_hyperparam_full_set, build_full_target_set
from datasets import Dataset, concatenate_datasets
from datasets.features import Image as HFImage
from omegaconf import OmegaConf, ListConfig
import random



@hydra.main(version_base=None, config_path="./config", config_name="run_img")
def main(cfg):
    
    print('''
          \n \n
          ==================================================
                            STARTING RUN 
          ==================================================
          \n \n
          '''
          )
    
    if cfg.job_meta_params.test_run:
        print('''
          \n \n
          ==================================================
                        !!THIS IS A TEST RUN!!
          ==================================================
          \n \n
          '''
          )
        print(f"Only Running on first {cfg.inference.test_number_of_batches} batches")
    
    print("Full Config Paramters:")
    print(cfg)
    print("\n")
    print("AUGMENTATIONS USED:")
    print(cfg.data.augmentations)
    print("\nREQUESTED DATA")
    
    
    # Load the target model
    target_model = load_target_model(cfg)

    # Generation data
    text = cfg.prompt.text
   
    print('''
          \n \n
          ==================================================
                            PREPARING DATA
          ==================================================
          \n \n
          '''
          )    
    
    _dataset = Dataset.from_parquet(cfg.data.dataset)
    
    
    _dataset = _dataset.add_column("indices", list(range(len(_dataset))))
    _dataset.to_json(os.path.join(cfg.path.output_dir,"full_dataset.json"))
    
    
    print("Sourcing Descriptions.....")
    if cfg.data.pre_gen_descriptions != "":
        print("Loading Descriptions from File...")
        with open(cfg.data.pre_gen_descriptions, "r") as f:
            descriptions = json.load(f)
            
    else:
        print("Generating Descriptions")
        # os.makedirs(os.path.join(cfg.path.output_dir, 'datasets'), exist_ok=True)
        descriptions = generate_descriptions(cfg, target_model, _dataset, out_path=os.path.join(cfg.path.output_dir, 'generated_descriptions.json'))
        
        
    #==============================
    # Dataset Summary Printout
    #===============================   
    print("Descriptions Loaded/Generated")
    print(f"descriptions length: {len(descriptions['sentences'])}")
    print(f"dataset length: {len(_dataset)}")
    # print(f'Number of true member samples: {_dataset["label"]}')
    # print(f'Number of true nonmember samples: {}')
    
    
    _dataset = _dataset.add_column("desc", descriptions["sentences"])
    
    with open(os.path.join(cfg.path.output_dir, "generated_sentences.json"), 'w', encoding='utf-8') as output_file:
        json.dump(descriptions, output_file, indent=4)
    
    
    print('''
          \n \n
          ==================================================
                            RUN COMPLETED
          ==================================================
          \n \n
          '''
          ) 

if __name__ == "__main__":
    main()