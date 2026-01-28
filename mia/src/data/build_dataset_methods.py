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
        member_target_set.append(sample)
    for idx in nonmem_idxs:
        sample = nonmember_data[idx]
        sample['tune_label'] = 1
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
