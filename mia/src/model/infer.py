import torch
from PIL import Image
from io import BytesIO
from src.model.utils import get_parts_slices
from minigpt.minigpt4.conversation.interact import Interact
from torchvision.transforms.functional import to_pil_image
import sys
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def _fast_stack_to_tensor(items, dtype):
    """
    Stacks a list of per-sample values into one batched tensor of the given dtype.

    HuggingFace `datasets` round-trips multi-dimensional values through Arrow, so by the time a
    `.map()`-produced field is read back via `dataset[idx]`, it comes back as a nested Python
    list, not the original torch.Tensor/np.ndarray -- even though it started as one inside the
    map function. `torch.tensor(nested_python_list)` builds the tensor by walking every scalar
    through the Python/C++ boundary one at a time, which is fine for a handful of values but very
    slow for e.g. a 336x336x3 image tensor repeated over a whole batch (profiled at ~21s/batch
    here). Converting through numpy first uses one fast bulk memory copy instead, then
    `torch.from_numpy` wraps that buffer with no further copying.

    Also handles the case where items are already tensors (e.g. if a future caller changes the
    upstream dataset format), by stacking directly rather than forcing an unnecessary
    list/numpy round-trip -- items are expected to already share one consistent shape either way,
    same as the torch.tensor(...)/torch.stack(...) calls this replaces required.
    """
    if len(items) > 0 and isinstance(items[0], torch.Tensor):
        return torch.stack(list(items)).to(dtype)
    np_dtype = {torch.float16: np.float16, torch.float32: np.float32, torch.int64: np.int64}[dtype]
    return torch.from_numpy(np.asarray(items, dtype=np_dtype)).to(dtype)


class BatchProcessor:
    def __init__(self, dataset, batch_size, eos_token_id, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % self.batch_size == 0 else int(len(dataset)/self.batch_size) + 1
        self.eos_token_id = eos_token_id

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.use_augmentation:
            return self._get_augmented_batch()    
        else:
            return self._get_normal_batch()

    def _get_normal_batch(self):
        """
        Do the manual padding & Attention masks
        Set the padding to the max-size of the input_ids (of current batch)
        """

        if self.current_batch == self.num_batch:
            raise StopIteration
 
        indices = list()
        input_ids = list()
        padded_input_ids = list()
        attention_masks = list()
        image_tensors = list()
        image_sizes = list()
        prompt_0 = list()
        prompt_1 = list()
        desc_shape = list()
        
        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size  
        
        input_ids_len = list()
        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            input_ids_len.append(len(self.dataset[_idx]["input_ids"]))
            image_tensors.append(self.dataset[_idx]["image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])
            prompt_0.append(self.dataset[_idx]["prompt_0"])
            prompt_1.append(self.dataset[_idx]["prompt_1"])
            desc_shape.append(self.dataset[_idx]["desc_shape"])

        max_length = max(input_ids_len)
        del input_ids_len

        for _idx in range(len(input_ids)):
            _input_ids = input_ids[_idx]
            padding_size = max_length - len(_input_ids)
            padding = self.eos_token_id * torch.ones(padding_size, dtype=torch.long)
            padded_input_id = torch.cat([torch.tensor(_input_ids, dtype=torch.long), padding])
            attention_mask = torch.zeros(max_length, dtype=torch.long)
            attention_mask[:len(_input_ids)] = 1
            padded_input_ids.append(padded_input_id)
            attention_masks.append(attention_mask)
        self.current_batch+=1

        return {
            "indices" : indices,
            "input_ids" : torch.stack(padded_input_ids, dim=0),
            "attention_masks" : torch.stack(attention_masks, dim=0),
            "image_sizes" :  torch.tensor(image_sizes),
            "image_tensors": torch.tensor(image_tensors, dtype=torch.float16),
            "prompt_0": prompt_0,
            "prompt_1": prompt_1,
            "desc_shape": desc_shape
        }

    def _get_augmented_batch(self):
        """
        Same: Do the manual padding & Attention masks
        Set the padding to the max-size of the current input_ids
        """

        if self.current_batch == self.num_batch:
            raise StopIteration
 
        indices = list()
        input_ids = list()
        padded_input_ids = list()
        attention_masks = list()
        orig_image_tensors = list()
        image_sizes = list()
        prompt_0 = list()
        prompt_1 = list()
        desc_shape = list()
        
        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size  
        
        input_ids_len = list()
        aug_image_tensors = dict()
        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            input_ids_len.append(len(self.dataset[_idx]["input_ids"]))
            orig_image_tensors.append(self.dataset[_idx]["orig_image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])
            prompt_0.append(self.dataset[_idx]["prompt_0"])
            prompt_1.append(self.dataset[_idx]["prompt_1"])
            desc_shape.append(self.dataset[_idx]["desc_shape"])

            for k, aug_imgs in self.dataset[_idx]["aug_image_tensors"].items():
                if k not in aug_image_tensors :
                    aug_image_tensors[k] = [[] for _ in range(len(aug_imgs))]
                for _aug_idx, _aug_img in enumerate(aug_imgs):
                    aug_image_tensors[k][_aug_idx].append(_aug_img)

        for k, aug_imgs in aug_image_tensors.items():
            for _aug_idx in range(len(aug_imgs)):
                aug_image_tensors[k][_aug_idx] = _fast_stack_to_tensor(aug_image_tensors[k][_aug_idx], torch.float16)

        max_length = max(input_ids_len)
        del input_ids_len

        for _idx in range(len(input_ids)):
            _input_ids = input_ids[_idx]
            padding_size = max_length - len(_input_ids)
            padding = self.eos_token_id * torch.ones(padding_size, dtype=torch.long)
            padded_input_id = torch.cat([torch.tensor(_input_ids, dtype=torch.long), padding])
            attention_mask = torch.zeros(max_length, dtype=torch.long)
            attention_mask[:len(_input_ids)] = 1
            padded_input_ids.append(padded_input_id)
            attention_masks.append(attention_mask)

        self.current_batch+=1

        return {
            "indices" : indices,
            "input_ids" : torch.stack(padded_input_ids, dim=0),
            "attention_masks" : torch.stack(attention_masks, dim=0),
            "image_sizes" :  _fast_stack_to_tensor(image_sizes, torch.int64),
            "orig_image_tensors": _fast_stack_to_tensor(orig_image_tensors, torch.float16),
            "aug_image_tensors": aug_image_tensors,
            "prompt_0": prompt_0,
            "prompt_1": prompt_1,
            "desc_shape": desc_shape
        }

class BatchProcessor_minigpt:
    def __init__(self, dataset, batch_size, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % self.batch_size == 0 else int(len(dataset)/self.batch_size) + 1

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_augmentation:
            return self._get_augmented_batch()
        else:
            return self._get_normal_batch()

    def _get_normal_batch(self):
        if self.current_batch == self.num_batch:
            raise StopIteration
        
        images = list()
        inst = list()
        desc = list()

        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size

        for _idx in range(batch_begin, batch_end):
            images.append(self.dataset[_idx]["raw_images"])
            inst.append(self.dataset[_idx]["inst"])
            desc.append(self.dataset[_idx]["desc"])
        self.current_batch+=1

        return {
            "images": images,
            "inst": inst,
            "desc": desc
        }

    def _get_augmented_batch(self):
        if self.current_batch == self.num_batch:
            raise StopIteration
        
        images = list()
        aug_images = list()
        inst = list()
        desc = list()

        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size

        for _idx in range(batch_begin, batch_end):
            _orig_img = self.dataset[_idx]["orig_images"]
            if isinstance(_orig_img, dict):
                _orig_img = Image.open(BytesIO(_orig_img["bytes"])).convert("RGB")
            images.append(_orig_img)
            _aug_img = self.dataset[_idx]["aug_images"]
            for k, v in _aug_img.items():
                for _vi in range(len(v)):
                    if isinstance(v[_vi], dict):
                        v[_vi] = Image.open(BytesIO(v[_vi]["bytes"])).convert("RGB")
            aug_images.append(_aug_img)
            inst.append(self.dataset[_idx]["inst"])
            desc.append(self.dataset[_idx]["desc"])             
        self.current_batch+=1

        return {
            "images": images,
            "aug_images": aug_images,
            "inst": inst,
            "desc": desc
        }
        
class BatchProcessor_hulu:
    def __init__(self, dataset, batch_size, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % self.batch_size == 0 else int(len(dataset)/self.batch_size) + 1

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_augmentation:
            return self._get_augmented_batch()
        else:
            return self._get_normal_batch()

    def _get_normal_batch(self):
        """
        Return the batched images and texts
        """

        if self.current_batch == self.num_batch:
            raise StopIteration
        
        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end= len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size

            images = list()
            inst = list()
            desc = list()
            
            for _idx in range(batch_begin, batch_end):
                images.append(self.dataset[_idx]["orig_raw_images"])
                inst.append(self.dataset[_idx]["inst"])
                desc.append(self.dataset[_idx]["desc"])
            self.current_batch+=1

            return {
                "images": images,
                "inst": inst,
                "desc": desc
            }
            
    def _get_augmented_batch(self):
            if self.current_batch == self.num_batch:
                raise StopIteration
            
            images = list()
            aug_images = list()
            inst = list()
            desc = list()

            batch_begin = self.current_batch * self.batch_size
            if self.current_batch == self.num_batch-1:
                batch_end = len(self.dataset)
            else:
                batch_end = batch_begin + self.batch_size

            for _idx in range(batch_begin, batch_end):
                sample_loaded_orig_imgs = list()
                sample_loaded_aug_imgs = dict()
                # The raw images of the sample
                _orig_imgs = self.dataset[_idx]["orig_raw_images"]
                for image in _orig_imgs:
                    if isinstance(image, dict):
                        image = Image.open(BytesIO(image["bytes"])).convert("RGB")
                    sample_loaded_orig_imgs.append(image)
                
                _aug_imgs = self.dataset[_idx]["aug_raw_images"]
                for aug_name, aug_matrix in _aug_imgs.items():
                    if aug_name not in sample_loaded_aug_imgs:
                        sample_loaded_aug_imgs[aug_name] = list()
                    for image_list in aug_matrix:
                        all_images_of_setting = list()
                        for image in image_list:
                            if isinstance(image, dict):
                                image = Image.open(BytesIO(image["bytes"])).convert("RGB")
                            all_images_of_setting.append(image)
                        sample_loaded_aug_imgs[aug_name].append(all_images_of_setting)
                        
                
                images.append(sample_loaded_orig_imgs)
                aug_images.append(sample_loaded_aug_imgs)
                inst.append(self.dataset[_idx]["inst"])
                desc.append(self.dataset[_idx]["desc"])             
            self.current_batch+=1

            return {
                "images": images,
                "aug_images": aug_images,
                "inst": inst,
                "desc": desc
            }
            
            
            
            
            

def mod_infer(model, dataset, cfg, tokenizer=None, vis_processor=None, chat_state=None, gpu_id=None):
    """
    Mod infer function
    Run the inference
    """
    if cfg.target_model.type == "llava":
        assert tokenizer == None
        batch_processor = BatchProcessor(dataset=dataset,
                                        batch_size=cfg.inference.batch_size,
                                        eos_token_id=tokenizer.eos_token_id,
                                        use_augmentation=cfg.inference.use_augmentation)
        
        all_results = []
        for b_idx, batch in enumerate(batch_processor):
            mix_input_ids, mix_attention_masks, target_parts = mod_infer_batch(
                model, batch, tokenizer, cfg.image_metrics.parts, cfg.inference.use_augmentation)
            all_results.append((mix_input_ids, mix_attention_masks, target_parts))

    elif cfg.target_model.type == "minigpt":
        assert vis_processor == None
        assert chat_state == None
        assert gpu_id == None
        batch_processor = BatchProcessor_minigpt(dataset=dataset,
                                                batch_size=cfg.inference.batch_size,
                                                use_augmentation=cfg.inference.use_augmentation)
        all_results = []
        for b_idx, batch in enumerate(batch_processor):
            mix_input_ids, mix_attention_masks, target_parts = mod_infer_batch_minigpt(
                model, vis_processor, batch, cfg.image_metrics.parts, chat_state, gpu_id, cfg.inference.use_augmentation)
            all_results.append((mix_input_ids, mix_attention_masks, target_parts))

    return all_results

def mod_infer_batch(model, batch, tokenizer, parts, use_augmentation):
    """
    mod_infer function.
    With the instruction, image and the descriptions,
    Do an inference using them.
    Return the batch meta metics (for all parts)

    model: target model
    batch: a batch of the dataloader from get_mod_infer_data
    tokenizer: tokenizer,
    use_augmentation : True if we use augmented
    """

    def _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape):
        target_parts = dict()
        labels_per_sample = list()
        for _input_ids, _logits, _attention_mask, _prompt_0, _prompt_1, _desc_shape \
            in zip(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape):

            _img_loss_slice, _img_slice, _inst_desc, _inst, _desc = get_parts_slices(_prompt_0, _prompt_1, _desc_shape)
            _img_loss_slices = _logits[_img_loss_slice, :]
            _img_target = torch.nn.functional.softmax(_img_loss_slices, dim=-1)
            _max_indices = torch.argmax(_img_target, dim=-1)

            # tensor a: Whatever that comes before the image
            # tensor b: From the second token after image to the end
            tensor_a = torch.tensor(_prompt_0).cuda() if not isinstance(_prompt_0, torch.Tensor) else _prompt_0
            tensor_b = torch.tensor(_prompt_1[1:]).cuda() if not isinstance(_prompt_1[1:], torch.Tensor) else _prompt_1[1:]

            _mix_input_ids = torch.cat([tensor_a, _max_indices, tensor_b], dim=0)

            for p in parts:
                if p not in target_parts:
                    target_parts[p] = {"input_ids": list(), "probabilities": list(), "log_probabilities": list()}
                if p == "img":
                    _slice = _img_slice
                elif p == "inst_desp":
                    _slice = _inst_desc
                elif p == "inst":
                    _slice = _inst
                elif p == "desp":
                    _slice = _desc
                else:
                    raise ValueError(f"Not supported goal {p}")

                target_parts[p]["input_ids"].append(_mix_input_ids[_slice])                
                _slice_logits = _logits[_slice, :]
                target_parts[p]["probabilities"].append(torch.nn.functional.softmax(_slice_logits, dim=-1))
                target_parts[p]["log_probabilities"].append(torch.nn.functional.log_softmax(_slice_logits, dim=-1))
                
            # Building Total Label
            labels = [''] * _mix_input_ids.shape[0]
            labels[_img_slice] = ['img']  * len(_mix_input_ids[_img_slice])
            labels[_inst]      = ['inst'] * len(_mix_input_ids[_inst])
            labels[_desc]      = ['desc'] * len(_mix_input_ids[_desc])
            labels_per_sample.append(labels)

        return target_parts, labels_per_sample
        

    if use_augmentation:
        input_ids = batch["input_ids"].cuda()
        attention_masks = batch["attention_masks"].cuda()
        orig_image_tensors = batch["orig_image_tensors"].cuda()
        image_sizes = batch["image_sizes"].cuda()

        prompt_0 = batch["prompt_0"]
        prompt_1 = batch["prompt_1"]
        desc_shape = batch["desc_shape"]
        
        total_parts = dict()
        total_token_labels = list()

        # 1. Conduct inference using the original images
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                images=orig_image_tensors,
                image_sizes=image_sizes
            )

        logits = outputs.logits
        target_parts, labels_per_sample = _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape)
        total_parts["orig"] = [target_parts]
        total_token_labels.extend(labels_per_sample)
        
        # 2. Conduct inference using the augmented images
        for k, aug_images in batch["aug_image_tensors"].items():
            total_parts[k] = list()
            for _aug_img in aug_images:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        images=_aug_img.cuda(),
                        image_sizes=image_sizes
                    )
                logits = outputs.logits
                target_parts, labels_per_sample = _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape)
                total_parts[k].append(target_parts)

        return total_parts, total_token_labels

    else:
        total_parts = dict()
        total_token_labels = list()
        

        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensors"].cuda()
        attention_masks = batch["attention_masks"].cuda()
        image_sizes = batch["image_sizes"].cuda()

        prompt_0 = batch["prompt_0"]
        prompt_1 = batch["prompt_1"]
        desc_shape = batch["desc_shape"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                images=image_tensors,
                image_sizes=image_sizes
            )

        logits = outputs.logits
        target_parts, labels_per_sample = _get_parts(input_ids, logits, attention_masks, prompt_0, prompt_1, desc_shape)
        total_parts["orig"] = [target_parts]
        total_token_labels.extend(labels_per_sample)

        return total_parts, total_token_labels

def mod_infer_batch_minigpt(model, vis_processor, batch, parts, chat_state, gpu_id, use_augmentation):

    def _get_parts_from_one_sample(input_ids, logits, seg_tokens, descp_encoding):
        """
        Slice the logit of one into different parts
        Input should only contain 1 sample
            - Consider implementing batched part processor in future (for the better efficiency)
        """
        target_parts = dict()
        labels_per_sample = list()

        _img_slice = slice(seg_tokens[0].shape[1],-seg_tokens[1].shape[1])
        _inst_desp = slice(-seg_tokens[-1].shape[1],None)
        _inst = slice(-seg_tokens[-1].shape[1],-descp_encoding.shape[1])
        _desp = slice(-descp_encoding.shape[1],None)

        img_loss_slice = logits[0, _img_slice.start-1:_img_slice.stop-1, :]
        img_target = torch.nn.functional.softmax(img_loss_slice, dim=-1)
        max_indices = torch.argmax(img_target, axis=-1)
        _mix_input_ids = torch.cat([seg_tokens[0][0], max_indices, seg_tokens[1][0]], dim=0)

        for p in parts:
            if p not in target_parts:
                target_parts[p] = { "input_ids": list(), "probabilities": list(), "log_probabilities": list()}
            if p == "img":
                _slice = _img_slice
            elif p == "inst_desp":
                _slice = _inst_desp
            elif p == "inst":
                _slice = _inst
            elif p == "desp":
                _slice = _desp
            else:
                raise ValueError(f"Not supported goal split {p}")

            target_parts[p]["input_ids"].append(_mix_input_ids[_slice])
            logits_slice = logits[0, _slice, :]
            
            target_parts[p]["probabilities"].append(
                torch.nn.functional.softmax(logits_slice, dim=-1)
            )
            target_parts[p]["log_probabilities"].append(
                torch.nn.functional.log_softmax(logits_slice, dim=-1)
            )

        # building total label
        labels = [''] * input_ids.shape[0]
        labels[_img_slice] = ['img'] * len(_mix_input_ids[_img_slice])
        labels[_inst] = ['inst'] * len(_mix_input_ids[_inst])
        labels[_desp] = ['desc'] * len(_mix_input_ids[_desp])
        labels_per_sample.append(labels)
        
        return target_parts, labels_per_sample

    total_parts = {
        "orig":  list()   
    }

    total_token_labels = list()

    if use_augmentation:
        images = batch["images"]
        aug_images = batch["aug_images"]
        inst = batch["inst"]
        desc = batch["desc"]

        # Serialized computation
        for _image, _aug_image, _inst, _desc in zip(images, aug_images, inst, desc):
            chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))
            _img_list = []
            _chat_state = chat_state.copy()

            # Make an inference on the augmented image            
            llm_message = chat.upload_img(_image, _chat_state, _img_list)
            chat.encode_img(_img_list)

            chat.ask(_inst, _chat_state)
            _chat_state.append_message(_chat_state.roles[1], None)
            _chat_state.append_message(_desc, None)

            outputs, input_ids, seg_tokens = chat.get_output_by_emb(
                conv=_chat_state,
                img_list = _img_list
            )

            desc_encoding = chat.model.llama_tokenizer(_desc, return_tensors="pt", add_special_tokens=False).to(chat.device).input_ids
            logits = outputs.logits
            target_parts, labels_per_sample = _get_parts_from_one_sample(input_ids, logits, seg_tokens, desc_encoding)

            if not len(total_parts["orig"]):
                total_parts["orig"].append(target_parts)
            else:
                for _part in parts:
                    for _key in total_parts["orig"][0][_part].keys():
                        total_parts["orig"][0][_part][_key].extend(target_parts[_part][_key])
            total_token_labels.extend(labels_per_sample)

            for k, aug_images in _aug_image.items():
                # For each augmentation type

                if k not in total_parts:
                    # Initialize the list for saving each setting
                    total_parts[k] = [None for _ in range(len(aug_images))]

                for _setting_idx, _aug_img in enumerate(aug_images):
                    # For each settings from the k-th augmentation
                    chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))
                    _img_list = []
                    _chat_state = chat_state.copy()    
                    print(k, _setting_idx, type(_aug_img))
                    llm_message = chat.upload_img(_aug_img, _chat_state, _img_list)
                    chat.encode_img(_img_list)

                    chat.ask(_inst, _chat_state)
                    _chat_state.append_message(_chat_state.roles[1], None)
                    _chat_state.append_message(_desc, None)

                    outputs, input_ids, seg_tokens = chat.get_output_by_emb(
                        conv=_chat_state,
                        img_list = _img_list
                    )
                    desc_encoding = chat.model.llama_tokenizer(_desc, return_tensors="pt", add_special_tokens=False).to(chat.device).input_ids
                    logits = outputs.logits
                    target_parts, labels_per_sample = _get_parts_from_one_sample(input_ids, logits, seg_tokens, desc_encoding)
                    if total_parts[k][_setting_idx] == None:
                        total_parts[k][_setting_idx] = target_parts
                    else:
                        for _part in parts:
                            # Insert (input_ids, probabilities, log_probabilities) from each part to the corresponding setting
                            for _key in total_parts[k][_setting_idx][_part].keys():
                                total_parts[k][_setting_idx][_part][_key].extend(target_parts[_part][_key]) 
        
        return total_parts, total_token_labels

    else:
        # No augmnentation
        
        total_parts = {
            "orig": []
        }
        total_token_labels = list()

        images = batch["images"]
        inst = batch["inst"]
        desc = batch["desc"]

        # Serialized computation
        # chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))
        for _image, _inst, _desc in zip(images, inst, desc):
            chat = Interact(model, vis_processor, device="cuda:{}".format(gpu_id))

            _img_list = []
            _chat_state = chat_state.copy()

            # Make an inference on the original image
            llm_message = chat.upload_img(_image, _chat_state, _img_list)
            chat.encode_img(_img_list)

            chat.ask(_inst, _chat_state)
            _chat_state.append_message(_chat_state.roles[1], None)
            _chat_state.append_message(_desc, None)

            outputs, input_ids, seg_tokens = chat.get_output_by_emb(
                conv=_chat_state,
                img_list = _img_list
            )

            desc_encoding = chat.model.llama_tokenizer(_desc, return_tensors="pt", add_special_tokens=False).to(chat.device).input_ids
            logits = output.logits
            target_parts, labels_per_sample = _get_parts_from_one_sample(input_ids, logits, seg_tokens, desc_encoding)

            if not len(total_parts["orig"]):
                total_parts["orig"].append(target_parts)
            else:
                for _key in total_parts["orig"][0].keys():
                    total_parts["orig"][0][_key].extend(target_parts[_key])
            total_token_labels.extend(labels_per_sample)

        return total_parts, total_token_labels
    














def mod_infer_batch_hulu(model, batch, tokenizer, vis_processor, parts, use_augmentation):
    def split_logit_by_parts(input_ids, logits, parts, image_token_id):
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1D [seq_len], got {tuple(input_ids.shape)}")
        if logits.dim() != 3:
            raise ValueError(f"logits must be 3D [batch, seq_len, vocab], got {tuple(logits.shape)}")
        if logits.size(0) < 1:
            raise ValueError("logits batch dimension is empty")
        if logits.size(1) != input_ids.numel():
            raise ValueError(f"seq_len mismatch: logits.size(1)={logits.size(1)} vs input_ids={input_ids.numel()}")

        seq_len = input_ids.numel()
        device = input_ids.device

        img_mask = (input_ids == image_token_id)
        img_slice = torch.nonzero(img_mask, as_tuple=False).squeeze(-1)
        inst_desc_slice = torch.nonzero(~img_mask, as_tuple=False).squeeze(-1)
        full_slice = torch.arange(seq_len, device=device)

        slices = {
            "img": img_slice,
            "inst_desc": inst_desc_slice,
            "img_inst_desc": full_slice,
        }

        # Build per-token labels (for this single sample)
        labels = ["inst_desc"] * seq_len
        for idx in img_slice.tolist():
            labels[idx] = "img"

        target_parts = {}
        for p in parts:
            if p not in slices:
                raise ValueError(f"Unsupported goal split {p}. Supported: {list(slices.keys())}")

            sl = slices[p]

            if p not in target_parts:
                target_parts[p] = {"input_ids": [], "probabilities": [], "log_probabilities": []}

            # ids for this slice
            target_parts[p]["input_ids"].append(input_ids[sl])

            # logits slice: [batch, seq_len, vocab] -> [L, vocab] for batch 0
            logits_slice = logits[0, sl, :]

            target_parts[p]["probabilities"].append(torch.nn.functional.softmax(logits_slice, dim=-1))
            target_parts[p]["log_probabilities"].append(torch.nn.functional.log_softmax(logits_slice, dim=-1))

        return target_parts, labels
        
    
    
    
    if use_augmentation:
        images_of_batch = batch["images"]
        aug_images_of_batch = batch["aug_images"]
        insts_of_batch = batch["inst"]
        descs_of_batch = batch["desc"]
        
        total_parts = {
            "orig":  list()   
        }

        total_token_labels = list()
        
        # Need to know the special token used for image, don't know if it violates blackbox
        image_token_id = vis_processor.image_token_id 
        

        for sample_images, sample_aug_images, sample_inst, sample_desc in zip(images_of_batch, aug_images_of_batch, insts_of_batch, descs_of_batch):            
            # Loop through and add each oriignal image to the conversation
            conversation = [
            {"role": "user", "content": []},   # images
            {"role": "user", "content": [{"type": "text", "text": sample_inst}]},
            {"role": "user", "content": [{"type": "text", "text": sample_desc}]},
            ]
    
            for _ in sample_images:
                conversation[0]["content"].append({"type": "image"})

            inputs = vis_processor(conversation=conversation, images=sample_images, return_tensors="pt")

            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
            with torch.inference_mode():
                out = model(**inputs, return_dict=True)
                logits = out.logits
                    
            input_ids = inputs["input_ids"][0]
            target_parts, labels = split_logit_by_parts(input_ids, logits, parts, image_token_id)    
            if not total_parts["orig"]:
                total_parts["orig"].append(target_parts)
            else:
                for _part in parts:
                    for _key in total_parts["orig"][0][_part].keys():
                        total_parts["orig"][0][_part][_key].extend(target_parts[_part][_key])
            total_token_labels.extend([labels])
            
            '''
            Of the form:
            
            {
                # aug1:
                [
                    # Setting 1
                    [img1, img2, img3],
                    # Setting 2
                    [img1, img2, img3]
                ],
                # aug2:
                [
                    # Setting 1
                    [img1, img2, img3],
                    # Setting 2
                    [img1, img2, img3]
                ]
            }
            '''
            
            for aug_name, aug_matrix in sample_aug_images.items():
                if aug_name not in total_parts:
                    # Initialize the list for saving each setting
                    total_parts[aug_name] = [None for _ in range(len(aug_matrix))]
                for setting_idx, setting_images in enumerate(aug_matrix):
                    conversation = [
                    {"role": "user", "content": []},   # images
                    {"role": "user", "content": [{"type": "text", "text": sample_inst}]},
                    {"role": "user", "content": [{"type": "text", "text": sample_desc}]},
                    ]
            
                    for _ in setting_images:
                        conversation[0]["content"].append({"type": "image"})
                        
                    inputs = vis_processor(conversation=conversation, images=setting_images, return_tensors="pt")
                    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                        
                    with torch.inference_mode():
                        out = model(**inputs, return_dict=True)
                        logits = out.logits
                            
                    input_ids = inputs["input_ids"][0]
                    target_parts, labels = split_logit_by_parts(input_ids, logits, parts, image_token_id)
                    
                    if total_parts[aug_name][setting_idx] == None:
                        total_parts[aug_name][setting_idx] = target_parts
                    else:
                        for _part in parts:
                            # Insert (input_ids, probabilities, log_probabilities) from each part to the corresponding setting
                            for _key in total_parts[aug_name][setting_idx][_part].keys():
                                total_parts[aug_name][setting_idx][_part][_key].extend(target_parts[_part][_key]) 
        
        return total_parts, total_token_labels
        
    else:
        raise NotImplementedError("Use Augmentation is False, not implimented yet")
        return None
        
                    
                    

        
            
            
            
            
# def sanity_check_split(input_ids, logits, target_parts, labels, image_token_id, parts):
#     seq_len = input_ids.numel()

#     # ---- Basic structure ----
#     assert isinstance(target_parts, dict), "target_parts must be a dict"
#     assert isinstance(labels, list), "labels must be a list"
#     assert len(labels) == seq_len, "labels length must equal seq_len"

#     for p in parts:
#         assert p in target_parts, f"Missing part: {p}"
#         entry = target_parts[p]
#         for k in ["input_ids", "probabilities", "log_probabilities"]:
#             assert k in entry, f"{p} missing key {k}"
#             assert len(entry[k]) == 1, f"{p}.{k} should have exactly one entry"

#     # ---- Image mask consistency ----
#     img_mask = (input_ids == image_token_id)
#     img_count = int(img_mask.sum().item())
#     rest_count = seq_len - img_count

#     # labels must agree with image mask
#     for i in range(seq_len):
#         if img_mask[i]:
#             assert labels[i] == "img", f"Label mismatch at pos {i}: expected 'img'"
#         else:
#             assert labels[i] == "inst_desc", f"Label mismatch at pos {i}: expected 'inst_desc'"

#     # ---- Slice lengths must match masks ----
#     if "img" in target_parts:
#         L_img = target_parts["img"]["input_ids"][0].numel()
#         assert L_img == img_count, f"img slice len {L_img} != img_count {img_count}"

#     if "inst_desc" in target_parts:
#         L_rest = target_parts["inst_desc"]["input_ids"][0].numel()
#         assert L_rest == rest_count, f"inst_desc slice len {L_rest} != rest_count {rest_count}"

#     if "img_inst_desc" in target_parts:
#         L_full = target_parts["img_inst_desc"]["input_ids"][0].numel()
#         assert L_full == seq_len, f"full slice len {L_full} != seq_len {seq_len}"

#     # ---- Logits alignment ----
#     vocab = logits.size(-1)
#     for p in parts:
#         probs = target_parts[p]["probabilities"][0]
#         log_probs = target_parts[p]["log_probabilities"][0]
#         ids = target_parts[p]["input_ids"][0]

#         assert probs.shape[0] == ids.numel(), f"{p} probs rows != ids length"
#         assert probs.shape[1] == vocab, f"{p} probs vocab mismatch"
#         assert log_probs.shape == probs.shape, f"{p} log_probs shape mismatch"

#     # ---- Probability sanity ----
#     # (softmax rows should sum to ~1)
#     # for p in parts:
#     #     probs = target_parts[p]["probabilities"][0]
#     #     if probs.numel() > 0:
#     #         row_sums = probs.sum(dim=-1)
#     #         assert torch.allclose(
#     #             row_sums,
#     #             torch.ones_like(row_sums),
#     #             atol=1e-4,
#     #         ), f"{p} probabilities do not sum to 1, sum is: {}"

#     print("✓ Sanity check passed")
            
            
            
            
            
            
            
              
            # # 1) image spans (one per <image> placeholder typically)
            
            # image_token_id = vis_processorcessor.image_token_id
            # img_spans = find_all_image_token_spans(input_ids, image_token_id)

            # # If you want "all images together" just merge them:
            # if len(img_spans) == 0:
            #     raise ValueError("No image token spans found (did the template include <image>?)")

            # img_span_all = (img_spans[0][0], img_spans[-1][1])

            # # 2) inst/desc body spans
            # inst_span = tagged_span(tokenizer, input_ids, "<INST>", "</INST>")
            # desc_span = tagged_span(tokenizer, input_ids, "<DESC>", "</DESC>")

            # # 3) slice logits aligned to each token region
            # img_ids,  img_logits  = slice_logits_for_tokens(input_ids, logits, img_span_all)
            # inst_ids, inst_logits = slice_logits_for_tokens(input_ids, logits, inst_span)
            # desc_ids, desc_logits = slice_logits_for_tokens(input_ids, logits, desc_span)
                
            # if len(total_parts["orig"]) == 0:
            #     total_parts["orig"].append(target_parts)
            # else:
            #     for _part in parts:
            #         for _key in total_parts["orig"][0][_part].keys():
            #             total_parts["orig"][0][_part][_key].extend(target_parts[_part][_key])

            
            
            # # 2. Conduct inference using the augmented images
            # for image_set in sample_aug_image.items():
            #     for k, aug_images in image_set.items():
            #             # For each augmentation type

            #             if k not in total_parts:
            #                 # Initialize the list for saving each setting
            #                 total_parts[k] = [None for _ in range(len(aug_images))]

            #             for _setting_idx, _aug_img in enumerate(aug_images):
                            
                            
            #                 conversation = [{
            #                     "role": "user",
            #                     "content": [
            #                         {"type": "image", "text_part": "image1", "data": None},
            #                         {"type": "text", "text_part": "inst", "text": "<INST>" + sample_inst + "</INST>"},
            #                         {"type": "text", "text_part": "desc", "text": "<DESC>" + sample_desc + "</DESC>"},
            #                     ]
            #                 }]
                            

                            
            #                 # Need to change so it accepts multiple images
            #                 # Format of encoder special tokens
            #                 prompt = (
            #                     "<IMG1>" + "<image>" + "</IMG1>\n"
            #                     "<INST>" + sample_inst + "</INST>\n"
            #                     "<DESC>" + sample_desc + "</DESC>"
            #                 )

            #                 inputs = vis_processor(images=[_aug_img], text=prompt, return_tensors="pt")
            #                 # n_img, (T,H,W), ms = n_img_from_inputs(inputs)
            #                 # print('AUG')
            #                 # print("PIL size:", (_aug_img.size if hasattr(_aug_img, "size") else type(_aug_img)))
            #                 # print("grid_sizes:", (T,H,W), "merge:", ms, "n_img:", n_img)
                            
            #                 # sys.exit()
                            
            #                 input_ids = inputs["input_ids"][0]
                            
            #                 # print("Printing input keys")
            #                 # print(inputs.keys())
                            
            #                 # print(f"Input_ids shape: {np.array(inputs['input_ids']).shape}")
                            
            #                 # Temp fix, not good
            #                 conversation[0]['content'][0]['data'] = _aug_img
                            
            #                 inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            #                 if "pixel_values" in inputs:
            #                     inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                                
            #                 with torch.inference_mode():
            #                     out = model(**inputs, return_dict=True)
            #                     logits = out.logits
                                
                            
                            
            #                 # print("Exiting")
            #                 # sys.exit()
            #                 # print(f"for aug: {k}")
            #                 target_parts, labels_per_sample = _get_parts_from_one_sample(conversation, prompt, input_ids, inputs, logits, tokenizer, vis_processor, parts, sample_inst, sample_desc)
                            
                            
            #                 if total_parts[k][_setting_idx] == None:
            #                     total_parts[k][_setting_idx] = target_parts
            #                 else:
            #                     for _part in parts:
            #                         # Insert (input_ids, probabilities, log_probabilities) from each part to the corresponding setting
            #                         for _key in total_parts[k][_setting_idx][_part].keys():
            #                             total_parts[k][_setting_idx][_part][_key].extend(target_parts[_part][_key]) 
            #     # print('exiting')
            #     # sys.exit()
                                        
            # # print(f"total_parts['org'] length: {len(total_parts['orig'])}")

        # return total_parts, total_token_labels    
    
    # else:
    #     raise NotImplementedError("Use Augmentation is False, not implimented yet")
    #     return None





def find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    """Return the start index of `needle` inside `haystack`, or -1."""
    if needle.numel() == 0 or haystack.numel() < needle.numel():
        return -1
    # sliding window compare
    for i in range(haystack.numel() - needle.numel() + 1):
        if torch.equal(haystack[i:i+needle.numel()], needle):
            return i
    return -1

def token_ids(text: str, tokenizer, input_ids) -> torch.Tensor:
    # IMPORTANT: add_special_tokens=False so we match literal markers
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)


















# def mod_infer_batch_hulu(model, batch, tokenizer, vis_processor, parts, use_augmentation):
#     """
#     HuluMed batch inference that deterministically splits token-level outputs into parts:
#       - img  : all <image> placeholder tokens (may be multiple contiguous runs; we keep them as a list)
#       - inst : tokens inside <INST>...</INST> (body only)
#       - desc : tokens inside <DESC>...</DESC> (body only)

#     Returns:
#       total_parts: dict mapping:
#         - "orig" -> aggregated dict of parts across samples
#         - each augmentation name -> list indexed by setting_idx, each entry aggregated across samples

#       total_token_labels: list (optional; currently empty but kept for compatibility)

#     Assumptions about `batch` when use_augmentation=True:
#       batch["images"]     : List[sample_images], where sample_images is List[image] (multi-image per sample OK)
#       batch["aug_images"] : List[sample_aug_images], where sample_aug_images is either:
#           (A) dict[str, dict[int, List[image]]]  e.g. {"gaussian": {0: [...imgs...], 1: [...imgs...]}, ...}
#           OR
#           (B) dict[str, List[List[image]]]       e.g. {"gaussian": [[...imgs...],[...imgs...]], ...}
#           OR
#           (C) dict[str, List[image]]             e.g. {"gaussian": [...imgs...] } (single setting)
#       batch["inst"]       : List[str]
#       batch["desc"]       : List[str]

#     Notes:
#       - Uses vis_processor as the HuluMedProcessor (AutoProcessor.from_pretrained(...)).
#       - Uses conversation=... and images=... (NOT text=conversation).
#       - Aligns logits to tokens via the standard causal-LM shift: logits[t] predicts input_ids[t+1].
#       - Stores probabilities and log_probabilities as requested:
#             probs = torch.softmax(logits_part, dim=-1)
#             log_probs = torch.log_softmax(logits_part, dim=-1)
#     """

#     # ---------------------- helpers: token spans ----------------------
#     def find_subseq(haystack, needle, start=0):
#         hay = haystack.tolist() if isinstance(haystack, torch.Tensor) else list(haystack)
#         ned = needle.tolist() if isinstance(needle, torch.Tensor) else list(needle)
#         n = len(ned)
#         for i in range(start, len(hay) - n + 1):
#             if hay[i : i + n] == ned:
#                 return (i, i + n)
#         return None

#     def tagged_span(tokenizer_, input_ids_, open_tag, close_tag):
#         ids = input_ids_.tolist() if isinstance(input_ids_, torch.Tensor) else list(input_ids_)

#         open_ids = tokenizer_.encode(open_tag, add_special_tokens=False)
#         close_ids = tokenizer_.encode(close_tag, add_special_tokens=False)

#         op = find_subseq(ids, open_ids, start=0)
#         if op is None:
#             raise ValueError(f"Open tag not found: {open_tag}")

#         cp = find_subseq(ids, close_ids, start=op[1])
#         if cp is None:
#             raise ValueError(f"Close tag not found: {close_tag}")

#         # body only (exclude the tags themselves)
#         return (op[1], cp[0])

#     def find_all_image_token_spans(input_ids_, image_token_id_):
#         """
#         Returns list of (start, end) runs where input_ids == image_token_id.
#         Multiple images can yield multiple runs (template-dependent).
#         """
#         ids = input_ids_.tolist() if isinstance(input_ids_, torch.Tensor) else list(input_ids_)
#         spans = []
#         i = 0
#         while i < len(ids):
#             if ids[i] == image_token_id_:
#                 j = i
#                 while j < len(ids) and ids[j] == image_token_id_:
#                     j += 1
#                 spans.append((i, j))
#                 i = j
#             else:
#                 i += 1
#         return spans

#     def slice_logits_for_tokens(input_ids_, logits_, token_span):
#         """
#         token_span: (s,e) over input_ids tokens you want to evaluate.
#         Returns:
#           token_ids   = input_ids[s:e]
#           token_logits= logits[s-1:e-1]  (aligned so each logit predicts the corresponding token)
#         """
#         if logits_.dim() == 3:
#             logits_2d = logits_[0]  # [seq, vocab]
#         else:
#             logits_2d = logits_

#         s, e = token_span
#         if s == 0:
#             s = 1  # cannot score token 0 (no previous logit predicts it)
#         if e <= s:
#             return input_ids_.new_empty((0,)), logits_2d.new_empty((0, logits_2d.size(-1)))

#         token_ids = input_ids_[s:e]
#         token_logits = logits_2d[s - 1 : e - 1, :]
#         return token_ids, token_logits

#     # ---------------------- helpers: accumulation ----------------------
#     def _init_aggregate(parts_list: List[str]) -> Dict[str, Dict[str, List[torch.Tensor]]]:
#         agg = {}
#         for p in parts_list:
#             agg[p] = {"input_ids": [], "probabilities": [], "log_probabilities": []}
#         return agg

#     def _append_part(agg, part_name: str, ids_tensor: torch.Tensor, logits_tensor: torch.Tensor):
#         # Store tensors (keep on GPU unless you want to .cpu() them)
#         agg[part_name]["input_ids"].append(ids_tensor)
#         probs = torch.softmax(logits_tensor, dim=-1)
#         log_probs = torch.log_softmax(logits_tensor, dim=-1)
#         agg[part_name]["probabilities"].append(probs)
#         agg[part_name]["log_probabilities"].append(log_probs)

#     def _merge_aggregate(dst, src):
#         # Extend lists
#         for p in src.keys():
#             for k in ["input_ids", "probabilities", "log_probabilities"]:
#                 dst[p][k].extend(src[p][k])

#     # ---------------------- helpers: normalize aug structure ----------------------
#     def normalize_aug_settings(sample_aug_images_obj):
#         """
#         Normalize augmented images for one sample into:
#           List[Tuple[aug_name, List[Tuple[setting_idx, images_list]]]]
#         where images_list is a List[image] (multi-image per sample OK).
#         Supports:
#           A) dict[str, dict[int, List[image]]]
#           B) dict[str, List[List[image]]]
#           C) dict[str, List[image]]  (single setting)
#         """
#         if sample_aug_images_obj is None:
#             return []

#         if not isinstance(sample_aug_images_obj, dict):
#             raise ValueError(f"aug_images per sample must be a dict, got: {type(sample_aug_images_obj)}")

#         out = []
#         for aug_name, aug_val in sample_aug_images_obj.items():
#             settings = []

#             # A) dict[int, List[image]]
#             if isinstance(aug_val, dict):
#                 for setting_idx in sorted(aug_val.keys()):
#                     imgs = aug_val[setting_idx]
#                     # allow single image (not list) -> wrap
#                     if imgs is None:
#                         continue
#                     if not isinstance(imgs, (list, tuple)):
#                         imgs = [imgs]
#                     settings.append((int(setting_idx), list(imgs)))

#             # B) list of settings, each is list of images
#             elif isinstance(aug_val, (list, tuple)):
#                 if len(aug_val) == 0:
#                     settings = []
#                 else:
#                     # If it's a "single setting" list-of-images (C), detect by element type not being list/tuple
#                     if not isinstance(aug_val[0], (list, tuple)):
#                         settings.append((0, list(aug_val)))
#                     else:
#                         for setting_idx, imgs in enumerate(aug_val):
#                             if imgs is None:
#                                 continue
#                             if not isinstance(imgs, (list, tuple)):
#                                 imgs = [imgs]
#                             settings.append((setting_idx, list(imgs)))
#             else:
#                 # single image
#                 settings.append((0, [aug_val]))

#             out.append((aug_name, settings))

#         return out

#     # ---------------------- core: infer one sample and split parts ----------------------
#     def infer_one(sample_images, sample_inst, sample_desc) -> Dict[str, Dict[str, List[torch.Tensor]]]:
#         """
#         Returns an aggregate dict with keys in `parts`.
#         For 'img', we aggregate across all image token runs.
#         For 'inst'/'desc', we aggregate across their body spans.
#         """
#         # Build conversation
#         conversation = [{"role": "user", "content": []}]
#         for _ in sample_images:
#             conversation[0]["content"].append({"type": "image"})
#         # combine inst+desc in one text chunk (reduces template-inserted separators)
#         conversation[0]["content"].append(
#             {"type": "text", "text": f"<INST>{sample_inst}</INST>\n<DESC>{sample_desc}</DESC>"}
#         )

#         # Processor call (IMPORTANT)
#         inputs = vis_processor(conversation=conversation, images=sample_images, return_tensors="pt")
#         input_ids = inputs["input_ids"][0]

#         # Move to CUDA
#         inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#         if "pixel_values" in inputs:
#             inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

#         with torch.inference_mode():
#             out = model(**inputs, return_dict=True)
#             logits = out.logits  # [1, seq, vocab]

#         # Sanity: align lengths
#         seq_len = int(input_ids.numel())
#         if logits.dim() != 3 or logits.size(1) != seq_len:
#             raise ValueError(f"Mismatch: logits={tuple(logits.shape)}, input_ids_len={seq_len}")

#         # Find spans
#         image_token_id = vis_processor.image_token_id
#         img_spans = find_all_image_token_spans(input_ids, image_token_id)
#         if len(img_spans) == 0:
#             raise ValueError("No image token spans found (template may not have inserted <image> tokens).")

#         inst_span = tagged_span(tokenizer, input_ids, "<INST>", "</INST>")
#         desc_span = tagged_span(tokenizer, input_ids, "<DESC>", "</DESC>")

#         # Build aggregate for this sample
#         agg = _init_aggregate(parts)

#         # Fill requested parts
#         for p in parts:
#             if p == "img":
#                 # aggregate across all image runs
#                 for sp in img_spans:
#                     ids_part, logits_part = slice_logits_for_tokens(input_ids, logits, sp)
#                     if ids_part.numel() > 0:
#                         _append_part(agg, "img", ids_part, logits_part)

#             elif p == "inst":
#                 ids_part, logits_part = slice_logits_for_tokens(input_ids, logits, inst_span)
#                 if ids_part.numel() > 0:
#                     _append_part(agg, "inst", ids_part, logits_part)

#             elif p in ["desc", "desp"]:
#                 ids_part, logits_part = slice_logits_for_tokens(input_ids, logits, desc_span)
#                 if ids_part.numel() > 0:
#                     _append_part(agg, p if p in agg else "desc", ids_part, logits_part)

#             elif p == "inst_desc" or p == "inst_desp":
#                 # body-only inst+desc, excluding the tags but including whatever lies between (newline)
#                 merged_span = (inst_span[0], desc_span[1])
#                 ids_part, logits_part = slice_logits_for_tokens(input_ids, logits, merged_span)
#                 if ids_part.numel() > 0:
#                     _append_part(agg, p, ids_part, logits_part)

#             else:
#                 raise ValueError(f"Unsupported part: {p}")

#         return agg

#     # ---------------------- main ----------------------
#     if not use_augmentation:
#         raise NotImplementedError("use_augmentation=False is not implemented yet.")

#     images_of_batch = batch["images"]
#     aug_images_of_batch = batch["aug_images"]
#     insts_of_batch = batch["inst"]
#     descs_of_batch = batch["desc"]

#     # total_parts["orig"] is ONE aggregate across the whole batch (same as your earlier style)
#     total_parts: Dict[str, Any] = {"orig": _init_aggregate(parts)}
#     total_token_labels: List[Any] = []

#     # For augmented: total_parts[aug_name][setting_idx] is an aggregate across the whole batch
#     # (created lazily when first seen)

#     for sample_images, sample_aug_images, sample_inst, sample_desc in zip(
#         images_of_batch, aug_images_of_batch, insts_of_batch, descs_of_batch
#     ):
#         # Normalize single-image -> list
#         if not isinstance(sample_images, (list, tuple)):
#             sample_images = [sample_images]

#         # ---- original ----
#         orig_agg = infer_one(sample_images, sample_inst, sample_desc)
#         _merge_aggregate(total_parts["orig"], orig_agg)

#         # ---- augmented ----
#         aug_norm = normalize_aug_settings(sample_aug_images)
#         for aug_name, settings in aug_norm:
#             if aug_name not in total_parts:
#                 # create list of aggregates indexed by setting_idx dynamically
#                 total_parts[aug_name] = {}

#             for setting_idx, aug_imgs in settings:
#                 if not isinstance(aug_imgs, (list, tuple)):
#                     aug_imgs = [aug_imgs]

#                 aug_agg = infer_one(list(aug_imgs), sample_inst, sample_desc)

#                 if setting_idx not in total_parts[aug_name]:
#                     total_parts[aug_name][setting_idx] = _init_aggregate(parts)

#                 _merge_aggregate(total_parts[aug_name][setting_idx], aug_agg)

#     # Optionally: convert per-aug dict-of-setting to list (dense) if you prefer
#     # Here we keep dict for safety in case settings are sparse.

#     return total_parts, total_token_labels














































