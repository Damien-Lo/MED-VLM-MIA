import os
import json
import torch
from pathlib import Path

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


class IncrementalRawMetaWriter:
    """
    Streams per-sample raw meta-metric values (e.g. full per-token probability
    distributions) to per-(part, metric_name, aug_name) fragment files as they're
    produced, instead of holding every sample for the whole run in RAM.

    finalize() assembles the fragments into a single JSON file shaped exactly like
    the old in-memory sampled_raw_meta:
        {part: {metric_name: {aug_name: [sample_entry, ...]}}}
    by streaming each fragment's lines straight into the output file, so peak
    memory during assembly is one sample at a time rather than the whole run.
    """

    def __init__(self, output_dir, filename="sampled_raw_meta"):
        os.makedirs(output_dir, exist_ok=True)
        self.final_path = os.path.join(output_dir, f"{filename}.json")
        self.tmp_dir = os.path.join(output_dir, f"_{filename}_fragments")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._handles = {}
        self._order = []

    def _fragment_path(self, part, metric_name, aug_name):
        return os.path.join(self.tmp_dir, f"{part}__{metric_name}__{aug_name}.jsonl")

    def _handle_for(self, part, metric_name, aug_name):
        key = (part, metric_name, aug_name)
        if key not in self._handles:
            self._handles[key] = open(self._fragment_path(*key), "w")
            self._order.append(key)
        return self._handles[key]

    def write_sample(self, part, metric_name, aug_name, value):
        f = self._handle_for(part, metric_name, aug_name)
        f.write(json.dumps(value))
        f.write("\n")

    def finalize(self):
        for f in self._handles.values():
            f.close()

        def _seen_in_order(items):
            seen = []
            for item in items:
                if item not in seen:
                    seen.append(item)
            return seen

        parts = _seen_in_order(part for part, _, _ in self._order)

        with open(self.final_path, "w") as out:
            out.write("{")
            for p_i, part in enumerate(parts):
                if p_i > 0:
                    out.write(",")
                out.write(json.dumps(part) + ":{")

                metric_names = _seen_in_order(mn for pt, mn, _ in self._order if pt == part)
                for m_i, metric_name in enumerate(metric_names):
                    if m_i > 0:
                        out.write(",")
                    out.write(json.dumps(metric_name) + ":{")

                    aug_names = [a for pt, mn, a in self._order if pt == part and mn == metric_name]
                    for a_i, aug_name in enumerate(aug_names):
                        if a_i > 0:
                            out.write(",")
                        out.write(json.dumps(aug_name) + ":[")

                        fragment_path = self._fragment_path(part, metric_name, aug_name)
                        with open(fragment_path) as fragment:
                            first_line = True
                            for line in fragment:
                                line = line.strip()
                                if not line:
                                    continue
                                if not first_line:
                                    out.write(",")
                                out.write(line)
                                first_line = False
                        out.write("]")
                        os.remove(fragment_path)

                    out.write("}")
                out.write("}")
            out.write("}")

        os.rmdir(self.tmp_dir)
        print(f"Saved sampled_raw_meta to {self.final_path}")


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
    
    txt = f"""\
        Description: {cfg.job_meta_params.description}
        TEST CASE: {cfg.job_meta_params.test_run}, Only Running on first {cfg.inference.test_number_of_batches} batches
        Run Job Type: {cfg.job_meta_params.job_type}
        Target Model: {cfg.target_model.type}
        Target Member Dataset: {cfg.data.member_dataset}
        Target Non-member Dataset: {cfg.data.nonmember_dataset}
        Reference Datasets Used: {cfg.data.reference_datasets_list}
        Reference Dataset Distribution: {cfg.data.reference_set_sample_distribution}
        Augmentations Used: {cfg.data.augmentations}
        Parts Tested: {cfg.img_metrics.parts}
        Metrics Tested: {cfg.img_metrics.metrics_to_use}

        Requested token labels of first {cfg.img_metrics.get_token_labels} of each class
        Requested Raw Augmented Images of first {cfg.img_metrics.get_raw_images} of each class
        print(f"Requested raw metrics values: {cfg.img_metrics.get_raw_meta_metrics}")
        print(f"Requested process metrics values: {cfg.img_metrics.get_raw_meta_metrics}")
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
        