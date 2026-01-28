import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets import Features, Value 
import os



def build_subset_json_file(data_dir, out_dir, set_size, membership_class):

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in data_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    subset = random.sample(images, min(set_size, len(images)))

    data = [{"image": str(p.resolve()), 'label': membership_class} for p in subset]

    with open(out_dir / f"member_ex{set_size}.json", "w") as f:
        json.dump(data, f, indent=2)

    return data





def main():
    print("Running")
    
    data_dir = '/local/scratch/clo37/datasets/PubMedVision/images'
    out_dir = '/local/scratch/clo37/datasets/PubMedVision/subsets'
    
    build_subset_json_file(data_dir,out_dir,300,1)
    
    
    print("Finished")


if __name__ == "__main__":
    main()
