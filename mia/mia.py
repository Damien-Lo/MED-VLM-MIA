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
from src.data.build_dataset_methods import build_target_set, json_to_dataset, list_to_dataset, get_random_subset, build_reference_set
from datasets import Dataset, concatenate_datasets
from datasets.features import Image as HFImage
from omegaconf import OmegaConf, ListConfig




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
    
    if cfg.img_metrics.get_token_labels > 0 :
        print(f"Requested token labels of first {cfg.img_metrics.get_token_labels} of each class")
        
    if cfg.img_metrics.get_raw_images > 0:
        print(f"Requested Raw Augmented Images of first {cfg.img_metrics.get_raw_images} of each class")
    
    if cfg.img_metrics.get_raw_meta_examples > 0:
        print(f"Requested metrics: {cfg.img_metrics.get_raw_meta_metrics} of first {cfg.img_metrics.get_raw_meta_examples} of each class")
        
    if cfg.img_metrics.get_proc_meta_examples > 0:
        print(f"Requested metrics: {cfg.img_metrics.get_proc_meta_metrics} of first {cfg.img_metrics.get_proc_meta_examples} of each class")
        
    save_run_meta(cfg)
        
        
    print('''
          \n \n
          ==================================================
                            LOADING MODEL
          ==================================================
          \n \n
          '''
          )

    # Load the target model
    target_model = load_target_model(cfg)

    # Generation data
    text = cfg.prompt.text
    # gen_path = os.path.join(os.getcwd(), "gen_descriptions", str(cfg.target_model.type), str(cfg.data.subset), "sentences.json")
    # with open(gen_path, 'r') as f:
    #     gen_data = json.load(f)
    # descriptions = gen_data["sentences"]

    
    # If we want to get meta values and labels for some samples (first x members and nonmembers) find the indecies these samples live
    print('''
          \n \n
          ==================================================
                            PREPARING DATA
          ==================================================
          \n \n
          '''
          )    
    
    if cfg.job_meta_params.job_type == "hyperparam_tuning":
        print("Hyperparamter selected, building combined target and reference set")
        
        #If member and non-member dataset paths are .json files, create DS object for each and combined to build the target set
        if os.path.splitext(cfg.data.member_dataset)[1].lower() == ".json" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".json":
            # Build a target set of members and nonmembers
            member_target_set, nm_target_set = build_target_set(cfg.data.target_set_size,
                                                                cfg.data.n_nm_ratio, 
                                                                cfg.data.member_dataset, 
                                                                cfg.data.nonmember_dataset)
            #Saving Parquet file of member target set
            list_to_dataset(member_target_set, out_path=(os.path.join(cfg.path.output_dir, "datasets", "member_target_dataset.parquet")))
            
            #Saving Parquet file of non-member target set
            target_non_member_set = list_to_dataset(nm_target_set,
                                                    out_path=(os.path.join(cfg.path.output_dir, "datasets", "non_member_target_dataset.parquet"))
                                                    )
            #Building combined target set from that instance
            target_set_list = member_target_set + nm_target_set
            target_dataset = list_to_dataset(target_set_list, out_path=(os.path.join(cfg.path.output_dir, "datasets", "target_dataset.parquet")))
            
        #If member and non-member dataset paths are .parquet files, just load them and concatinate them directly
        elif os.path.splitext(cfg.data.member_dataset)[1].lower() == ".parquet" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".parquet":
            target_member_dataset = Dataset.from_parquet(cfg.data.member_dataset)
            target_non_member_set = Dataset.from_parquet(cfg.data.nonmember_dataset)
            target_dataset = concatenate_datasets([target_member_dataset, target_non_member_set])
        
        #Else, through error
        else:
            target_dataset = None
            target_non_member_set = None
            raise ValueError("Member or Non-member dataset not valid type (json or parquet)")
            
        # Building Reference Set of known non_members equal to size of target set from a distribution/list of possible non_member sources
        # If a list of dataset paths are given build a combined reference set from weighted distirbution of list options
        if isinstance(cfg.data.reference_datasets_list,ListConfig):
            reference_datasets_paths = list(cfg.data.reference_datasets_list)
            reference_datasets_distribution = list(cfg.data.reference_set_sample_distribution)
            reference_dataset = build_reference_set(cfg.data.target_set_size, reference_datasets_paths, reference_datasets_distribution)
            #If reference dataset is not empty, build reference set accordingly and save as parquet file
            if reference_dataset != None:
                reference_dataset = list_to_dataset(reference_dataset, out_path=(os.path.join(cfg.path.output_dir, "datasets", "reference_dataset.parquet") if cfg.data.save_datasets else None))
            #Otherwise just use the exact nonmember as reference set
            else:
                print("No Reference Set Given, using exact non_member set as referece set too")
                reference_dataset = copy.deepcopy(target_non_member_set).remove_columns("tune_label")
                if cfg.data.save_datasets:
                    reference_dataset.to_parquet(os.path.join(cfg.path.output_dir, "datasets", "reference_dataset.parquet"))
        #If refefence set is not a list and a just a parquet file, load it directly
        else:
            reference_dataset = Dataset.from_parquet(cfg.data.reference_datasets_list)
            
        # Add the tune label of 0 to the reference set
        reference_dataset = reference_dataset.add_column('tune_label', [0]*len(reference_dataset))
        _dataset = concatenate_datasets([target_dataset,reference_dataset])
        # Save Full Dataset
        _dataset.to_parquet(os.path.join(cfg.path.output_dir, "datasets", "full_dataset.parquet"))
    elif cfg.job_meta_params.job_type == "evaluation":
        print("Evaluation selected, loading singlar target dataset from file")
        _dataset = Dataset.from_parquet(cfg.data.dataset)
    else:
        raise ValueError(f"'{cfg.job_meta_params.job_type}' is not a valid job_type, please use either 'hyperparam_tuning' or 'evaluation'. ")
    
    
    # _dataset = _dataset.add_column("indices", list(range(len(_dataset)))).cast_column("image", HFImage(decode=True))
    _dataset = _dataset.add_column("indices", list(range(len(_dataset))))
    _dataset.to_json(os.path.join(cfg.path.output_dir,"full_dataset.json"))
    
    
    print("Sourcing Descriptions.....")
    # if cfg.data.pre_gen_descriptions != "" or "desc" not in [p for p in cfg.img_metrics.parts]:
    if cfg.data.pre_gen_descriptions != "":
        print("Loading Descriptions from File or Skipping description gen as no description required")
        with open(cfg.data.pre_gen_descriptions, "r") as f:
            descriptions = json.load(f)
        # descriptions = {"sentences": [""] * len(_dataset)}
        # member_idxs, nonmember_idxs, descriptions = build_descriptions_dataset(cfg)
    else:
        print("Generating Descriptions")
        descriptions = generate_descriptions(cfg, _dataset, out_path=None)
        
    print("Descriptions Loaded/Generated")
    print(f"descriptions length: {len(descriptions['sentences'])}")
    print(f"dataset length: {len(_dataset)}")
    
    _dataset = _dataset.add_column("desc", descriptions["sentences"])
    
    # with open(os.path.join(cfg.path.output_dir, "datasets", "generated_sentences.json"), 'w', encoding='utf-8') as output_file:
    #     json.dump(descriptions, output_file, indent=4)


    print("Generating Inference and Augmentations.....")
    if cfg.target_model.type == "llava":
        model, tokenizer, image_processor, conv_mode = target_model
        mod_infer_data, image_sampled_indicies = get_mod_infer_data(cfg, text, _dataset, model.config, tokenizer, image_processor, conv_mode)
    elif cfg.target_model.type == "minigpt":
        mod_infer_data, image_sampled_indicies = get_mod_infer_data(cfg, text, _dataset)
    elif cfg.target_model.type == "hulu_med":
        mod_infer_data, image_sampled_indicies = get_mod_infer_data(cfg, text, _dataset)
        
        
    proc_meta_vaues_sampled_indices = list()
    raw_meta_vaues_sampled_indices = list()
    
    # Class Labels
    true_class_labels = mod_infer_data["label"]
    labels_to_save = {"true_class_labels" : true_class_labels}
    if cfg.job_meta_params.job_type == "hyperparam_tuning":
        labels_to_save['tune_class_labels'] = _dataset['tune_label']
        
    
    
    if cfg.job_meta_params.test_run:
        true_class_labels = true_class_labels[: (cfg.inference.batch_size * cfg.inference.test_number_of_batches)]
    

    if cfg.img_metrics.get_raw_meta_examples > 0:
        raw_meta_vaues_sampled_indices = np.sort(np.concatenate([
        np.where(np.array(true_class_labels) == 1)[0][:cfg.img_metrics.get_raw_meta_examples],
        np.where(np.array(true_class_labels) == 0)[0][:cfg.img_metrics.get_raw_meta_examples]
        ]))
        
        
    if cfg.img_metrics.get_proc_meta_examples > 0:
        proc_meta_vaues_sampled_indices = np.sort(np.concatenate([
        np.where(np.array(true_class_labels) == 1)[0][:cfg.img_metrics.get_proc_meta_examples],
        np.where(np.array(true_class_labels) == 0)[0][:cfg.img_metrics.get_proc_meta_examples]
        ]))
    print("Completed. Tokens Acquired")
    
    print(f"Raw Meta values sampled Indecies: {raw_meta_vaues_sampled_indices}")
    print(f"Processed Meta values sampled Indecies: {proc_meta_vaues_sampled_indices}")
    
    

    # Get the Raw Original Image and Augment Tensor Image
    if len(mod_infer_data['orig_raw_images']) > 0:
        print('''
          \n \n
          ==================================================
                    SAVING RAW IMAGES TENSORS.....
          ==================================================
          \n \n
          '''
        )
        
        
        save_stack = list()
        for img in mod_infer_data['orig_raw_images']:
            if img != None:
                save_stack.append(img)
        save_to_pt(save_stack, "orig_image_tensors", cfg)
        
        save_stack = list()
        for img in mod_infer_data['aug_raw_images']:
            if img !=None:
                save_stack.append(img)
        save_to_pt(save_stack, "aug_image_tensors", cfg)
    
        print("RAW IMAGE SAVE COMPLETE")   
    if cfg.img_metrics.get_raw_images > 0:
        save_to_json(image_sampled_indicies, "image_sampled_indicies", cfg)
        
        
    print('''
          \n \n
          ==================================================
                        BEGINNING INFERENCE
          ==================================================
          \n \n
          '''
          )
    
    if cfg.target_model.type == "llava":
        model, tokenizer, image_processor, conv_mode = target_model
        preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, mod_infer_data, raw_meta_vaues_sampled_indices, proc_meta_vaues_sampled_indices, cfg, tokenizer=tokenizer)
    elif cfg.target_model.type == "minigpt":
        model, vis_encoder, chat_state = target_model
        gpu_id = model.device.index if hasattr(model, "device") and hasattr(model.device, "index") else 0
        preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, mod_infer_data, raw_meta_vaues_sampled_indices, proc_meta_vaues_sampled_indices,
            cfg, vis_processor=vis_encoder, gpu_id=gpu_id, chat_state=chat_state)
    elif cfg.target_model.type == "hulu_med":
        model, tokenizer, image_processor = target_model
        preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, mod_infer_data, raw_meta_vaues_sampled_indices, proc_meta_vaues_sampled_indices, cfg, tokenizer=tokenizer, vis_processor=image_processor)
        

    print('''
          \n \n
          ==================================================
                        INFERENCE COMPLETE
          ==================================================
          \n \n
          '''
          )
    
    print("Saving preds to json....")
    save_to_json(preds, "preds", cfg)
    
    if cfg.img_metrics.get_token_labels > 0:
        print("Saving token labels to json...")
        save_to_json(proc_meta_vaues_sampled_indices.tolist(), "all_proc_meta_sampled_examples",cfg)
        save_to_json(list(true_class_labels), "class_labels", cfg)
        save_to_json(global_token_labels, "token_labels", cfg)
        
    if cfg.img_metrics.get_raw_meta_examples > 0:
        print("Saving raw meta values to pt......")
        save_to_json(raw_meta_vaues_sampled_indices.tolist(), "all_raw_meta_sampled_examples",cfg)
        save_to_json(list(true_class_labels), "class_labels", cfg)
        save_to_pt(sampled_raw_meta, "raw_meta_values", cfg)
    
    if cfg.img_metrics.get_proc_meta_examples > 0:
        print("Saving processed meta values to json....")
        save_to_json(proc_meta_vaues_sampled_indices.tolist(), "all_proc_meta_sampled_examples",cfg)
        save_to_json(list(true_class_labels), "class_labels", cfg)
        save_to_json(proc_meta, "processed_meta_values", cfg)
        
    print('''
          \n \n
          ==================================================
                        BEGINNING EVALUATION
          ==================================================
          \n \n
          '''
          )    
    # Evaluation
    if cfg.job_meta_params.job_type == "hyperparam_tuning":
        auc, acc, auc_low = evaluate(preds, mod_infer_data["tune_label"], "img", cfg) 
    elif cfg.job_meta_params.job_type == "evaluation":
       auc, acc, auc_low = evaluate(preds, mod_infer_data["label"], "img", cfg) 
    else:
        raise ValueError(f"'{cfg.job_meta_params.job_type}' is not a valid job_type, please use either 'hyperparam_tuning' or 'evaluation'. ")       

    # Save
    print("Saving evaluation results.....")
    save_to_json(auc, "auc", cfg)
    save_to_json(acc, "acc", cfg)
    save_to_json(auc_low, "tpr_005_fpr", cfg)
    
    
    if cfg.job_meta_params.job_type == "hyperparam_tuning":
         print('''
          \n \n
          ======================================================================
                            Finding sigma of minimal kld difference
          ======================================================================
          \n \n
          '''
          ) 
    
    
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