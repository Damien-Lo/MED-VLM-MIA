import os
import json
import sys
from tqdm import tqdm
from src.model import mod_infer_batch, mod_infer_batch_minigpt, mod_infer_batch_hulu
from src.model.infer import BatchProcessor, BatchProcessor_minigpt, BatchProcessor_hulu
from src.metrics import get_meta_metrics_by_part, get_img_metric_by_parts
import numpy as np
from collections import defaultdict
import torch

def inference(model, dataset, meta_sampled_indices, cfg,
              tokenizer=None, vis_processor=None, gpu_id=None, chat_state=None):
    """
    For each batch
        1. Conduct mod-infer
        2. Obtain meta-metrics
        3. Obtain img_metrics
        4. Combine img_metrics
    """

    if cfg.target_model.type == "llava":
        assert tokenizer != None
        batch_processor = BatchProcessor(dataset=dataset,
                                        batch_size=cfg.inference.batch_size,
                                        eos_token_id=tokenizer.eos_token_id,
                                        use_augmentation=cfg.inference.use_augmentation)
    elif cfg.target_model.type == "minigpt":
        assert vis_processor != None
        assert chat_state != None
        assert gpu_id != None 
        batch_processor = BatchProcessor_minigpt(dataset=dataset,
                                                batch_size=cfg.inference.batch_size,
                                                use_augmentation=cfg.inference.use_augmentation)
    elif cfg.target_model.type == "hulu_med":
        assert tokenizer != None
        assert vis_processor != None
        
        batch_processor = BatchProcessor_hulu(dataset=dataset,
                                                 batch_size=cfg.inference.batch_size,
                                                 use_augmentation=cfg.inference.use_augmentation)
        
    else:
        raise ValueError(f"Unknown model type: {cfg.target_model.type}")

    parts = cfg.img_metrics.parts
    global_pred = dict()
    sampled_proc_meta = dict()
    sampled_raw_meta = dict()
    global_token_labels = list()
    
    
    for _part in parts:
        global_pred[_part] = dict()
        sampled_proc_meta[_part] = dict()
        sampled_raw_meta[_part] = dict()
        
        
    meta_sample_map = defaultdict(list)
    for idx in meta_sampled_indices:
        meta_sample_map[int(idx/cfg.inference.batch_size)].append(idx % cfg.inference.batch_size)
    print(f"Meta Sample Map: {meta_sample_map}")

        
    
    for b_idx, batch in enumerate(tqdm(batch_processor,
                                       total=len(batch_processor),
                                       desc="Running inference",
                                       unit="batch")):
            
        # For Test Run, just run 1 batch
        if cfg.job_meta_params.test_run and b_idx >= cfg.inference.test_number_of_batches:
            break
            
        if cfg.target_model.type == "llava":
            target_parts, total_token_labels = mod_infer_batch(model, batch, tokenizer,
                                        parts=parts,
                                        use_augmentation=cfg.inference.use_augmentation)
        elif cfg.target_model.type == "minigpt":
            target_parts, total_token_labels = mod_infer_batch_minigpt(
                model, vis_processor, batch, parts, chat_state, gpu_id, cfg.inference.use_augmentation)
            
        elif cfg.target_model.type == "hulu_med":
            target_parts, total_token_labels = mod_infer_batch_hulu(
                model, batch, tokenizer, vis_processor, parts, cfg.inference.use_augmentation)


        else:
            raise ValueError(f"Unknown model type {cfg.target_model.type}")
        
        global_token_labels.extend(total_token_labels)
        
        # Process separately for each part
        for _part in parts:
            _meta_metrics = get_meta_metrics_by_part(target_parts,
                                                     part=_part,
                                                    cfg=cfg)
            
            if b_idx in meta_sample_map:
                for metric_name, aug_dict in _meta_metrics.items():
                    if metric_name not in cfg.img_metrics.get_raw_meta_metrics:
                        continue

                    if metric_name not in sampled_raw_meta[_part]:
                        sampled_raw_meta[_part][metric_name] = dict()

                    for aug_name, value_array in aug_dict.items():
                        if aug_name not in sampled_raw_meta[_part][metric_name]:
                            sampled_raw_meta[_part][metric_name][aug_name] = list()

                        for sample_idx in meta_sample_map[b_idx]:
                            per_sample_all_settings = []
                            for setting_idx, setting in enumerate(value_array):
                                per_sample_all_settings.append(np.array(setting[sample_idx]).tolist())
                            sampled_raw_meta[_part][metric_name][aug_name].append(per_sample_all_settings)
                        

                
            
            # Get Processed Meta Values        
            _pred, _proc_meta = get_img_metric_by_parts(_meta_metrics, cfg)         
            
            #============================
            # Storing _pred scores
            #============================
            # If first iteration and global_pred is empty, just add batch's metrics
            if len(global_pred[_part]) == 0:
                global_pred[_part] = _pred
            else:
                for metric_category, metrics in _pred.items():
                    for metric_name, metric_values in metrics.items():
                        if metric_category in set(['kld_metrics', 'renyi_div_metrics']):
                            for aug, aug_settings in metric_values.items():
                                for aug_setting, scores in aug_settings.items():
                                    global_pred[_part][metric_category][metric_name][aug][aug_setting].extend(scores)
                        elif metric_category in set(['baseline_metrics']):
                            if metric_name in set(['aug_kl']):
                                global_pred[_part][metric_category][metric_name].extend(metric_values)
                            elif metric_name in set(['min_k', 'min_k_renyi_05_entro', 'min_k_renyi_1_entro','mink', 'max_k_renyi_1_entro', 'max_k_renyi_05_entro']):
                                for setting, scores in metric_values.items():
                                    global_pred[_part][metric_category][metric_name][setting].extend(scores)
                            else:
                                raise ValueError(f"Unknown baseline metric {metric_name}")
                        else:
                            raise ValueError(f"Unknown metric category {metric_category}")               
                
            #===================================
            # Storing processed meta values
            #===================================
            # Final Structure of Meta Values:
            '''
            {
                part: {
                    metric_category: {
                        metric_name: {
                            aug_name:{
                                aug_setting_params: [sample][kl_values]  # 3D array
                            }
                        }
                    }
                }
            }
            '''
            if b_idx in meta_sample_map and cfg.img_metrics.get_meta_examples >= 0:
                for metric_category, metrics in _proc_meta.items():
                    if metric_category not in sampled_proc_meta[_part]:
                        sampled_proc_meta[_part][metric_category] = dict()
                    for metric_name, meta_dict in metrics.items():
                        if metric_name not in sampled_proc_meta[_part][metric_category]:
                            sampled_proc_meta[_part][metric_category][metric_name] = dict()
                        # if metric_category in set(['kld_metrics', 'renyi_div_metrics']):
                        for aug, settings in meta_dict.items():
                            if aug not in sampled_proc_meta[_part][metric_category][metric_name]:
                                sampled_proc_meta[_part][metric_category][metric_name][aug] = dict()
                            for setting_params, raw_values in settings.items():
                                if setting_params not in sampled_proc_meta[_part][metric_category][metric_name][aug]:
                                    sampled_proc_meta[_part][metric_category][metric_name][aug][setting_params] = list()
                                for sample_idx, sample_klds in enumerate(raw_values):
                                    if sample_idx in meta_sample_map[b_idx]:
                                        if isinstance(sample_klds, np.ndarray): 
                                            sample_klds = sample_klds.tolist()
                                        sampled_proc_meta[_part][metric_category][metric_name][aug][setting_params].append(sample_klds)       
            
                
            # for metric_name, metric_values in _pred.items():
            #     if metric_name not in global_pred[_part]:
            #         # Initialize the dictionary
            #         if isinstance(metric_values, list):
            #             global_pred[_part][metric_name] = list()
            #         elif isinstance(metric_values, dict):
            #             global_pred[_part][metric_name] = dict()
            #             for metric_key in metric_values.keys():
            #                 global_pred[_part][metric_name][metric_key] = list()
            #     if isinstance(metric_values, list):
            #         global_pred[_part][metric_name].extend(metric_values)
            #     elif isinstance(metric_values, dict):
            #         for _key, _value in metric_values.items():
            #             global_pred[_part][metric_name][_key].extend(_value)
                        
                    
            # Final Structure of Meta Values:
            '''
            {
                part: {
                    metric_name: {
                        aug_title:
                            [setting][sample][kl_values]  # 3D array
                    }
                }
            }
            '''
            # if b_idx in proc_meta_sample_map and cfg.img_metrics.get_meta_examples >= 0:
            #     for metric_name, meta_dict in _proc_meta.items():
            #         if metric_name not in sampled_proc_meta[_part]:
            #                 sampled_proc_meta[_part][metric_name] = defaultdict(list)
                    
            #         for aug, meta in meta_dict.items():
            #             for sample_idx, kld_for_all_settings in enumerate(meta):
            #                 if sample_idx in proc_meta_sample_map[b_idx]:
            #                     sampled_proc_meta[_part][metric_name][aug].append(kld_for_all_settings)
                    
                    
                            
                    
     
    # If We want want the meta values, only return the values at the desired sampled_indices, else return empty array         
    # if cfg.img_metrics.get_token_labels > 0 or cfg.img_metrics.get_proc_meta_values > 0:
    #     global_token_labels = [global_token_labels[i] for i in proc_meta_sampled_indices]
                
    if cfg.img_metrics.get_token_labels <= 0 or cfg.img_metrics.get_meta_examples <= 0:
        global_token_labels = []
        sampled_proc_meta = {}
        
        
    # for part, metric_dict in sampled_proc_meta.items():
    #     for metric_name, metric_values in metric_dict.items():
    #         for aug, value_array in metric_values.items():
    #             if isinstance(value_array, np.ndarray):
    #                 sampled_proc_meta[part][metric_name][aug] = value_array.tolist()
        
    return global_pred, sampled_raw_meta, sampled_proc_meta, global_token_labels 
