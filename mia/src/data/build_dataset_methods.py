import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets import Features, Value  # (optional)
import os
from datasets.features import Image as HFImage
import math
import sys
from itertools import zip_longest

# Helper Functions
def random_int_list(m: int, n: int):
    if m > n + 1:
        raise ValueError("m cannot exceed the size of the range (n+1).")
    return random.sample(range(n), m)

def ratios_to_units(ratios, num_of_units):
    if num_of_units < 0:
        raise ValueError("set_size must be non-negative")
    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")
    
    raw = [r * num_of_units for r in ratios]
    floors = [math.floor(x) for x in raw]
    counts = floors[:]

    leftover = num_of_units - sum(floors)
    if leftover > 0:
        remainders = [x - f for x, f in zip(raw, floors)]
        order = sorted(range(len(ratios)), key=lambda i: (-remainders[i], i))
        for i in order[:leftover]:
            counts[i] += 1

    return counts

def get_random_subset(data, num_of_samp):
    idxs = random_int_list(num_of_samp, len(data))
    result = list()
    
    for idx in idxs:
        result.append(data[idx])
        
    return result

# Main Functions
def build_target_set(set_size, m_nm_ratio, member_data_path, non_member_data_path):
    with open(member_data_path, "r") as f:
        member_data = json.load(f)
    with open(non_member_data_path, "r") as f:
        nonmember_data = json.load(f)
    
    
    num_of_mem = int(set_size * m_nm_ratio)
    num_of_nonmem = set_size - num_of_mem
    
    if num_of_mem > len(member_data):
        raise IndexError(f"Given ratio and set size, target member samples of {num_of_mem} exceeds member data length {len(member_data)}")
    
    if num_of_nonmem > len(nonmember_data):
         raise IndexError(f"Given ratio and set size, target nonmember samples of {num_of_nonmem} exceeds nonmember data length {len(nonmember_data)}")
    
    mem_idxs = random_int_list(num_of_mem, len(member_data))
    nonmem_idxs = random_int_list(num_of_nonmem, len(nonmember_data))
    
    member_target_set = list()
    nm_target_set = list()
    
    for idx in mem_idxs:
        sample = member_data[idx]
        sample['tune_label'] = 1
        sample['label'] = 1
        member_target_set.append(sample)
    for idx in nonmem_idxs:
        sample = nonmember_data[idx]
        sample['tune_label'] = 1
        sample['label'] = 0
        nm_target_set.append(sample)
    
    return member_target_set, nm_target_set


def build_reference_set(set_size, dataset_path_list, dataset_sample_ratios):
    if len(dataset_path_list) == 0:
        return None
    
    if len(dataset_sample_ratios) == 0 or sum(dataset_sample_ratios) != 1:
        raise ValueError("Sum of given dataset sample ratios do not sum up to 1")
    
    reference_set = list()
    samples_to_take = ratios_to_units(dataset_sample_ratios, set_size)
    
    for idx, data_path in enumerate(dataset_path_list):
        print(data_path)
        with open(data_path, "r") as f:
            data = json.load(f)
        reference_set.extend(get_random_subset(data,samples_to_take[idx]))
    return reference_set

def json_to_dataset(json_path, out_path=None):

    ds = load_dataset("json", data_files=json_path)

    ds = ds.cast_column("image", HFImage(decode=True))
    
    if out_path != None:
        ds.to_parquet(out_path)
        
    return ds

def list_to_dataset(list, out_path=None):
    # ds = Dataset.from_list(list).cast_column("image", HFImage(decode=True))
    ds = Dataset.from_list(list)
    
    if out_path != None:
        print("Saving Parquet")
        ds.to_parquet(out_path)
        
    return ds

def interleaf_sets(member_target_set, nm_target_set):
    """
    Alternate items from member_target_set and nm_target_set.
    If one is longer, append the remainder at the end.
    """
    out = []
    for m, nm in zip_longest(member_target_set, nm_target_set, fillvalue=None):
        if m is not None:
            out.append(m)
        if nm is not None:
            out.append(nm)
    return out



#=============================================
# FINAL BUILD METHODS
#=============================================

def build_hyperparam_full_set(cfg):
    #============================================================
    #                       BUILD TARGET SETS
    #============================================================
    #==============================
    # JSON Source Sets
    #==============================
    #If member and non-member dataset paths are .json files, create DS object for each and combined to build the target set
    if os.path.splitext(cfg.data.member_dataset)[1].lower() == ".json" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".json":
        member_target_set, nm_target_set = build_target_set(cfg.data.target_set_size,
                                                            cfg.data.n_nm_ratio, 
                                                            cfg.data.member_dataset, 
                                                            cfg.data.nonmember_dataset)
        list_to_dataset(member_target_set, out_path=(os.path.join(cfg.path.output_dir, "datasets", "member_target_dataset.parquet")))
        target_non_member_set = list_to_dataset(nm_target_set,out_path=(os.path.join(cfg.path.output_dir, "datasets", "non_member_target_dataset.parquet"))) #target_non_member_set saved for reference set if none is given
        target_set_list = interleaf_sets(member_target_set,nm_target_set)
        target_dataset = list_to_dataset(target_set_list, out_path=(os.path.join(cfg.path.output_dir, "datasets", "target_dataset.parquet")))
    #==============================
    # PARQUET Source Sets
    #==============================
    elif os.path.splitext(cfg.data.member_dataset)[1].lower() == ".parquet" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".parquet":
        target_member_dataset = Dataset.from_parquet(cfg.data.member_dataset)
        target_non_member_set = Dataset.from_parquet(cfg.data.nonmember_dataset)
        target_dataset = concatenate_datasets([target_member_dataset, target_non_member_set])
    #==============================
    # OTHERWISE ERROR
    #==============================
    else:
        target_dataset = None
        target_non_member_set = None
        raise ValueError("Member or Non-member dataset not valid type (json or parquet)")
    
    
    #============================================================
    #                       BUILD REFERENCE SETS
    #============================================================
    #==============================
    # NO Reference Set Given
    #==============================
    if (cfg.data.reference_datasets_list == None or
        cfg.data.reference_datasets_list == '' or
        (isinstance(cfg.data.reference_datasets_list,ListConfig) and len(cfg.data.reference_datasets_list) == 0)
        ):
        print("No Reference Set Given, using exact non_member set as referece set too")
        reference_dataset = copy.deepcopy(target_non_member_set).remove_columns("tune_label")
    
    #==============================
    # Only one reference set given
    #==============================
    #TODO: Fix, I don't think this is so right
    elif (
        (isinstance(cfg.data.reference_datasets_list, str) and os.path.splitext(cfg.data.reference_datasets_list)[1].lower() == ".json") or
        (isinstance(cfg.data.reference_datasets_list,ListConfig) and len(cfg.data.reference_datasets_list) == 1)
    ):
        reference_dataset = list_to_dataset(reference_dataset, out_path=(os.path.join(cfg.path.output_dir, "datasets", "reference_dataset.parquet") if cfg.data.save_datasets else None))
    elif(
        (isinstance(cfg.data.reference_datasets_list, str) and os.path.splitext(cfg.data.reference_datasets_list)[1].lower() == ".parquet")
    ):
        reference_dataset = Dataset.from_parquet(cfg.data.reference_datasets_list)
    
    elif (
        (isinstance(cfg.data.reference_datasets_list,ListConfig) and len(cfg.data.reference_datasets_list) > 1)
    ):
        reference_datasets_paths = list(cfg.data.reference_datasets_list)
        reference_datasets_distribution = list(cfg.data.reference_set_sample_distribution)
        reference_dataset = build_reference_set(cfg.data.target_set_size, reference_datasets_paths, reference_datasets_distribution)
    else:
        raise ValueError('Reference set source format not valid, please either give json, parquet files or lists of json file paths.')
    
    reference_dataset = reference_dataset.add_column('tune_label', [0]*len(reference_dataset))
    
    #============================================================
    #                 BUILD FINAL FULL DATASET
    #============================================================
    _dataset = concatenate_datasets([target_dataset,reference_dataset])
    _dataset.to_parquet(os.path.join(cfg.path.output_dir, "datasets", "full_dataset.parquet"))
    
    return _dataset



def build_full_target_set(cfg):
    #============================================================
    #                       BUILD TARGET SETS
    #============================================================
    #==============================
    # JSON Source Sets
    #==============================
    #If member and non-member dataset paths are .json files, create DS object for each and combined to build the target set
    if os.path.splitext(cfg.data.member_dataset)[1].lower() == ".json" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".json":
        member_target_set, nm_target_set = build_target_set(cfg.data.target_set_size,
                                                            cfg.data.n_nm_ratio, 
                                                            cfg.data.member_dataset, 
                                                            cfg.data.nonmember_dataset)
        list_to_dataset(member_target_set, out_path=(os.path.join(cfg.path.output_dir, "datasets", "member_target_dataset.parquet")))
        target_non_member_set = list_to_dataset(nm_target_set,out_path=(os.path.join(cfg.path.output_dir, "datasets", "non_member_target_dataset.parquet"))) #target_non_member_set saved for reference set if none is given
        target_set_list = interleaf_sets(member_target_set, nm_target_set)
        with open(os.path.join(cfg.path.output_dir, "datasets", "target_dataset.json"), "w") as f:
            json.dump(target_set_list, f, indent=2)
        target_dataset = list_to_dataset(target_set_list, out_path=(os.path.join(cfg.path.output_dir, "datasets", "target_dataset.parquet")))
    #==============================
    # PARQUET Source Sets
    #==============================
    elif os.path.splitext(cfg.data.member_dataset)[1].lower() == ".parquet" and os.path.splitext(cfg.data.nonmember_dataset)[1].lower() == ".parquet":
        target_member_dataset = Dataset.from_parquet(cfg.data.member_dataset)
        target_non_member_set = Dataset.from_parquet(cfg.data.nonmember_dataset)
        target_dataset = concatenate_datasets([target_member_dataset, target_non_member_set]).to_parquet(os.path.join(cfg.path.output_dir, "datasets", "target_dataset.parquet"))
    #==============================
    # OTHERWISE ERROR
    #==============================
    else:
        target_dataset = None
        target_non_member_set = None
        raise ValueError("Member or Non-member dataset not valid type (json or parquet)")
    
    return target_dataset