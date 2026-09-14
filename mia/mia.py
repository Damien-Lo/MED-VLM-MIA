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
from src.eval import evaluate, compute_tuning_separation, slice_preds
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

    # Reproducibility seed -- set before ANYTHING else in this run that could touch randomness
    # (dataset building's member/nonmember/reference sampling for job_type=build_dataset, or the
    # Gaussian noise augmentation draw for hyperparam_tuning/evaluation/tune_and_eval), so it's
    # guaranteed to cover whichever this particular `python mia.py` invocation actually does.
    # cfg.job_meta_params.seed is null by default (generate a fresh one); pass an explicit value
    # to reproduce a specific past run -- the value actually used is always saved to
    # run_parameters.txt below, whether that was auto-generated or user-supplied.
    if cfg.job_meta_params.seed is None:
        cfg.job_meta_params.seed = random.randint(0, 2**31 - 1)
    seed = cfg.job_meta_params.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print('''
          \n \n
          ==================================================
                            STARTING RUN
          ==================================================
          \n \n
          '''
          )
    print(f"Reproducibility seed for this run: {seed}\n")

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
    
    if cfg.img_metrics.get_meta_examples > 0:
        print(f"Requested raw metrics values: {cfg.img_metrics.get_raw_meta_metrics}")
        print(f"Requested process metrics values: {cfg.img_metrics.get_raw_meta_metrics}")
        print(f"Will get {cfg.img_metrics.get_meta_examples} or maximum of member and nonmember lengths")
        
    
    
    # Called before the build_dataset branch (which exits early) too, not just before the MIA
    # run below -- run_parameters.txt (seed included) should exist for every `python mia.py`
    # invocation, dataset-building ones included, not just hyperparam_tuning/evaluation/
    # tune_and_eval. This used to only run for the latter, so a build_dataset run's own seed
    # (independent from whatever MIA run later consumes its output) was never recorded at all.
    save_run_meta(cfg)

    if cfg.job_meta_params.job_type == "build_dataset":
        print('''
          \n \n
          ==================================================
                JUST BUILDING A TARGET SET REQUESTED
          ==================================================
          \n \n
          '''
          )
        build_full_target_set(cfg)
        print('Target Dataset Built and saved')
        sys.exit()

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
        print("Hyperparamter selected, building combined target and reference set....")    
        _dataset = build_hyperparam_full_set(cfg)
        print('Hyperparam dataset sucessfully built and saved \n')
        
        # #If member and non-member dataset paths are .json files, create DS object for each and combined to build the target set
        # if os.path.splitext(cfg.data.member_dataset)[1].lower() == ".json" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".json":
        #     # Build a target set of members and nonmembers
        #     member_target_set, nm_target_set = build_target_set(cfg.data.target_set_size,
        #                                                         cfg.data.n_nm_ratio, 
        #                                                         cfg.data.member_dataset, 
        #                                                         cfg.data.nonmember_dataset)
        #     #Saving Parquet file of member target set
        #     list_to_dataset(member_target_set, out_path=(os.path.join(cfg.path.output_dir, "datasets", "member_target_dataset.parquet")))
            
        #     #Saving Parquet file of non-member target set
        #     target_non_member_set = list_to_dataset(nm_target_set,
        #                                             out_path=(os.path.join(cfg.path.output_dir, "datasets", "non_member_target_dataset.parquet"))
        #                                             )
        #     #Building combined target set from that instance
        #     target_set_list = member_target_set + nm_target_set
        #     target_dataset = list_to_dataset(target_set_list, out_path=(os.path.join(cfg.path.output_dir, "datasets", "target_dataset.parquet")))
            
        # #If member and non-member dataset paths are .parquet files, just load them and concatinate them directly
        # elif os.path.splitext(cfg.data.member_dataset)[1].lower() == ".parquet" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".parquet":
        #     target_member_dataset = Dataset.from_parquet(cfg.data.member_dataset)
        #     target_non_member_set = Dataset.from_parquet(cfg.data.nonmember_dataset)
        #     target_dataset = concatenate_datasets([target_member_dataset, target_non_member_set])
        
        # #Else, through error
        # else:
        #     target_dataset = None
        #     target_non_member_set = None
        #     raise ValueError("Member or Non-member dataset not valid type (json or parquet)")
            
        # # Building Reference Set of known non_members equal to size of target set from a distribution/list of possible non_member sources
        # # If a list of dataset paths are given build a combined reference set from weighted distirbution of list options
        # if isinstance(cfg.data.reference_datasets_list,ListConfig):
        #     reference_datasets_paths = list(cfg.data.reference_datasets_list)
        #     reference_datasets_distribution = list(cfg.data.reference_set_sample_distribution)
        #     reference_dataset = build_reference_set(cfg.data.target_set_size, reference_datasets_paths, reference_datasets_distribution)
        #     #If reference dataset is not empty, build reference set accordingly and save as parquet file
        #     if reference_dataset != None:
        #         reference_dataset = list_to_dataset(reference_dataset, out_path=(os.path.join(cfg.path.output_dir, "datasets", "reference_dataset.parquet") if cfg.data.save_datasets else None))
        #     #Otherwise just use the exact nonmember as reference set
        #     else:
        #         print("No Reference Set Given, using exact non_member set as referece set too")
        #         reference_dataset = copy.deepcopy(target_non_member_set).remove_columns("tune_label")
        #         if cfg.data.save_datasets:
        #             reference_dataset.to_parquet(os.path.join(cfg.path.output_dir, "datasets", "reference_dataset.parquet"))
        # #If refefence set is not a list and a just a parquet file, load it directly
        # else:
        #     reference_dataset = Dataset.from_parquet(cfg.data.reference_datasets_list)
            
        # # Add the tune label of 0 to the reference set
        # reference_dataset = reference_dataset.add_column('tune_label', [0]*len(reference_dataset))
        # _dataset = concatenate_datasets([target_dataset,reference_dataset])
        # # Save Full Dataset
        # _dataset.to_parquet(os.path.join(cfg.path.output_dir, "datasets", "full_dataset.parquet"))
    
    elif cfg.job_meta_params.job_type == "evaluation":
        print("Evaluation selected, loading singlar target dataset from file...")
        _dataset = Dataset.from_parquet(cfg.data.dataset)
        print('Evaluation dataset sucessfully loaded from parquet \n')
    elif cfg.job_meta_params.job_type == "tune_and_eval":
        print("Tune-and-eval selected, loading pre-built target and reference datasets from file...")
        # Both datasets are pre-built and frozen (see build_and_save_reference_set /
        # build_full_target_set) -- no resampling here, unlike hyperparam_tuning. Target rows
        # are placed first in the concatenation so they can be recovered later with a plain
        # slice (see target_set_size below / slice_preds), without a second inference pass.
        target_dataset = Dataset.from_parquet(cfg.data.dataset)
        reference_dataset = Dataset.from_parquet(cfg.data.reference_dataset)
        target_set_size = len(target_dataset)
        _dataset = concatenate_datasets([target_dataset, reference_dataset])
        print(f"Tune-and-eval dataset loaded: {target_set_size} target-set rows + {len(reference_dataset)} reference-set rows \n")
    else:
        raise ValueError(f"'{cfg.job_meta_params.job_type}' is not a valid job_type, please use either 'hyperparam_tuning', 'evaluation', or 'tune_and_eval'. ")
    
    
    _dataset = _dataset.add_column("indices", list(range(len(_dataset))))
    _dataset.to_json(os.path.join(cfg.path.output_dir,"full_dataset.json"))
    
    
    print("Sourcing Descriptions.....")
    # if cfg.data.pre_gen_descriptions != "" or "desc" not in [p for p in cfg.img_metrics.parts]:
    if cfg.data.pre_gen_descriptions != "":
        print("Loading Descriptions from File...")
        with open(cfg.data.pre_gen_descriptions, "r") as f:
            descriptions = json.load(f)
        # descriptions = {"sentences": [""] * len(_dataset)}
        # member_idxs, nonmember_idxs, descriptions = build_descriptions_dataset(cfg)
    else:
        print("Generating Descriptions")
        os.makedirs(os.path.join(cfg.path.output_dir, 'datasets'), exist_ok=True)
        descriptions = generate_descriptions(cfg, target_model, _dataset, out_path=os.path.join(cfg.path.output_dir, 'datasets', 'generated_descriptions.json'))
        
        
    #==============================
    # Dataset Summary Printout
    #===============================   
    print("Descriptions Loaded/Generated")
    print(f"descriptions length: {len(descriptions['sentences'])}")
    print(f"dataset length: {len(_dataset)}")
    # print(f'Number of true member samples: {_dataset["label"]}')
    # print(f'Number of true nonmember samples: {}')
    
    
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
        
        
    proc_meta_values_sampled_indices = list()
    raw_meta_values_sampled_indices = list()
    
    # Class Labels
    true_class_labels = mod_infer_data["label"]
        
    
    
    if cfg.job_meta_params.test_run:
        true_class_labels = true_class_labels[: (cfg.inference.batch_size * cfg.inference.test_number_of_batches)]
        
    labels_to_save = {"true_class_labels" : true_class_labels}
    if cfg.job_meta_params.job_type in ("hyperparam_tuning", "tune_and_eval"):
        labels_to_save['tune_class_labels'] = _dataset['tune_label']
    if cfg.job_meta_params.job_type == "tune_and_eval":
        # class_labels.json should describe just the target set here, matching the shape
        # job_type=evaluation produces (preds.json/auc.json below are sliced to target-only
        # too) -- tune_class_labels is kept full-length (target+reference) since that's what's
        # needed later to split target-vs-reference for the tuning outputs.
        labels_to_save['true_class_labels'] = true_class_labels[:target_set_size]

    #TODO: Need to fix, somehow make the sampling better, ensure the labels are assigned, returned and printed correctly
    # and try to only sample indeciies you really want
    # For evaluation, I obviously want to sample by members and non-members, for hyperparam turning, I want to sample by tune_class_labels member and nonmembers
    # In such a case, as the distribution of the dataset "members" and "non_members" could be different, I need to cap
    # the return based on the smaller dataset

    member_idxs = np.where(np.array(true_class_labels) == 1)[0]
    non_member_idxs = np.where(np.array(true_class_labels) == 0)[0]

    if cfg.job_meta_params.job_type == "tune_and_eval":
        # Restrict raw/proc meta sampling to target-set rows only (indices 0..target_set_size-1,
        # since target rows are always placed first in the tune_and_eval concatenation), so
        # sampled_raw_meta/proc_meta stay target-set-only -- matching job_type=evaluation's
        # output instead of mixing in reference-set rows.
        member_idxs = member_idxs[member_idxs < target_set_size]
        non_member_idxs = non_member_idxs[non_member_idxs < target_set_size]


    if cfg.img_metrics.get_meta_examples > 0:
        samples_to_take = min(cfg.img_metrics.get_meta_examples, len(member_idxs), len(non_member_idxs))
        print(f'Taking the minium of requested, number of members or number of non_members: {samples_to_take} samples')
        random_idx_sample = np.random.permutation(np.arange(0, samples_to_take))
        meta_sampled_indices = np.sort(np.concatenate([
            member_idxs[random_idx_sample],
            non_member_idxs[random_idx_sample]
        ]))
        
        
        save_to_json(labels_to_save, "class_labels", cfg)
        
    print(f"Sampling these indecies for the request raw and processed meta values: {meta_sampled_indices}, saving all class_labels")
    
    

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
        preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, mod_infer_data, meta_sampled_indices, cfg, tokenizer=tokenizer)
    elif cfg.target_model.type == "minigpt":
        model, vis_encoder, chat_state = target_model
        gpu_id = model.device.index if hasattr(model, "device") and hasattr(model.device, "index") else 0
        preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, mod_infer_data, meta_sampled_indices,
            cfg, vis_processor=vis_encoder, gpu_id=gpu_id, chat_state=chat_state)
    elif cfg.target_model.type == "hulu_med":
        model, tokenizer, hulu_processor = target_model
        preds, sampled_raw_meta, proc_meta, global_token_labels = inference(model, mod_infer_data, meta_sampled_indices, cfg, tokenizer=tokenizer, vis_processor=hulu_processor)
        

    print('''
          \n \n
          ==================================================
                        INFERENCE COMPLETE
          ==================================================
          \n \n
          '''
          )
    
    if cfg.job_meta_params.job_type == "tune_and_eval":
        # preds.json is written further down, sliced to target-set-only, so it matches the
        # shape job_type=evaluation produces instead of the full target+reference combination.
        print("Skipping raw preds.json save here (tune_and_eval writes a target-set-only preds.json below)....")
    else:
        print("Saving preds to json....")
        save_to_json(preds, "preds", cfg)

    print("Saving sampled_raw_meta to json....")
    save_to_json(sampled_raw_meta, "sampled_raw_meta", cfg)
    
    print("Saving proc_meta to json....")
    save_to_json(proc_meta, "proc_meta", cfg)
        
    print(''')
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
    elif cfg.job_meta_params.job_type == "tune_and_eval":
        # --- Tuning half: target-set vs reference-set divergence across the full noise range ---
        print("Computing tuning separation (target-set vs reference-set) across the full noise range....")
        tuning_raw_split, tuning_separation, tuning_best_by_setting = compute_tuning_separation(
            preds, mod_infer_data["tune_label"], cfg
        )
        print("Saving tuning outputs to json....")
        save_to_json(tuning_raw_split, "tuning_raw_scores", cfg)
        save_to_json(
            {"separation": tuning_separation, "best_by_setting": tuning_best_by_setting},
            "tuning_separation", cfg
        )

        # --- Eval half: target-set-only member vs non-member, same shape as job_type=evaluation ---
        print("Slicing target-set-only predictions for the real membership eval....")
        target_preds = slice_preds(preds, target_set_size)
        print("Saving preds to json....")
        save_to_json(target_preds, "preds", cfg)
        auc, acc, auc_low = evaluate(target_preds, mod_infer_data["label"][:target_set_size], "img", cfg)
    else:
        raise ValueError(f"'{cfg.job_meta_params.job_type}' is not a valid job_type, please use either 'hyperparam_tuning', 'evaluation', or 'tune_and_eval'. ")

    # Save
    print("Saving evaluation results.....")
    save_to_json(auc, "auc", cfg)
    save_to_json(acc, "acc", cfg)
    save_to_json(auc_low, "tpr_low_fpr", cfg)
    
    
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