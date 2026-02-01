import torch
from PIL import Image
from io import BytesIO
from src.model.utils import get_parts_slices
from minigpt.minigpt4.conversation.interact import Interact
from torchvision.transforms.functional import to_pil_image
import sys
import numpy as np


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
                aug_image_tensors[k][_aug_idx] = torch.tensor(aug_image_tensors[k][_aug_idx], dtype=torch.float16)

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
            "orig_image_tensors": torch.tensor(orig_image_tensors, dtype=torch.float16),
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
                _orig_img = self.dataset[_idx]["orig_raw_images"]
                if isinstance(_orig_img, dict):
                    _orig_img = Image.open(BytesIO(_orig_img["bytes"])).convert("RGB")
                images.append(_orig_img)
                _aug_img = self.dataset[_idx]["aug_raw_images"]
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
    



# def mod_infer_batch_hulu(model, batch, tokenizer, vis_processor, parts, use_augmentation):
#     print("Begining mod_infer_batch_hulu")

#     def _get_parts_from_one_sample(conversation, prompt, input_ids, inputs, logits,
#                                    tokenizer, vis_processor, parts, sample_inst, sample_desc):

#         def visual_token_count_from_inputs(inputs, image_index) -> int:
#             gs = inputs["grid_sizes"][image_index]     # [T,H,W] or [1,H,W]
#             ms = inputs["merge_sizes"][image_index]    # scalar
#             if hasattr(ms, "item"):
#                 ms = int(ms.item())
#             T, H, W = [int(x) for x in gs.tolist()]
#             return (T if T > 1 else 1) * (H // ms) * (W // ms)

#         def find_runs(haystack, token_id):
#             runs = []
#             i, n = 0, len(haystack)
#             while i < n:
#                 if haystack[i] == token_id:
#                     j = i + 1
#                     while j < n and haystack[j] == token_id:
#                         j += 1
#                     runs.append((i, j))
#                     i = j
#                 else:
#                     i += 1
#             return runs

#         def find_subseq(haystack, needle, start=0, end=None):
#             if end is None:
#                 end = len(haystack)
#             m = len(needle)
#             if m == 0:
#                 return (start, start)
#             for i in range(start, end - m + 1):
#                 if haystack[i:i + m] == needle:
#                     return (i, i + m)
#             return None

#         def get_spans_multi_image(input_ids, inputs, tokenizer, vis_processor, sample_inst, sample_desc):
#             seq = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
#             image_token_id = vis_processor.image_token_id

#             # expected placeholder counts per image
#             n_images = len(inputs["grid_sizes"])
#             expected = [visual_token_count_from_inputs(inputs, k) for k in range(n_images)]

#             # actual runs of image placeholder tokens
#             runs = find_runs(seq, image_token_id)
#             if len(runs) == 0:
#                 raise ValueError("No image_token_id run found in input_ids")
#             if len(runs) < n_images:
#                 # common edge case: processor may merge all images into one long run
#                 # If there is a single run, split it using expected lengths.
#                 if len(runs) == 1 and n_images > 1:
#                     s0, e0 = runs[0]
#                     total_expected = sum(expected)
#                     if (e0 - s0) < total_expected:
#                         raise ValueError(
#                             f"Single image run length {e0-s0} < sum(expected) {total_expected}. "
#                             f"expected={expected}"
#                         )
#                     image_spans = []
#                     cur = s0
#                     for exp_len in expected:
#                         image_spans.append((cur, cur + exp_len))
#                         cur += exp_len
#                 else:
#                     raise ValueError(f"Found {len(runs)} image-token runs, expected {n_images}")
#             else:
#                 # Greedy match runs to expected lengths in order
#                 image_spans = []
#                 run_idx = 0
#                 for k, exp_len in enumerate(expected):
#                     while run_idx < len(runs) and (runs[run_idx][1] - runs[run_idx][0]) != exp_len:
#                         run_idx += 1
#                     if run_idx >= len(runs):
#                         raise ValueError(
#                             f"Could not match image {k} with expected len {exp_len}. "
#                             f"expected={expected}, runs={[r[1]-r[0] for r in runs]}"
#                         )
#                     image_spans.append(runs[run_idx])
#                     run_idx += 1

#             # Search for inst/desc after last image span (fallback to whole seq if needed)
#             text_search_start = image_spans[-1][1]

#             inst_ids = tokenizer.encode(sample_inst, add_special_tokens=False)
#             desc_ids = tokenizer.encode(sample_desc, add_special_tokens=False)

#             inst_span = find_subseq(seq, inst_ids, start=text_search_start)
#             if inst_span is None:
#                 inst_span = find_subseq(seq, inst_ids, start=0)
#             if inst_span is None:
#                 raise ValueError("Could not locate instruction text tokens in input_ids")

#             desc_span = find_subseq(seq, desc_ids, start=inst_span[1])
#             if desc_span is None:
#                 desc_span = find_subseq(seq, desc_ids, start=0)
#             if desc_span is None:
#                 raise ValueError("Could not locate description text tokens in input_ids")

#             return image_spans, inst_span, desc_span

#         # ---- use new span function here ----
#         seq = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
#         image_spans, inst_span, desc_span = get_spans_multi_image(
#             input_ids, inputs, tokenizer, vis_processor, sample_inst, sample_desc
#         )

#         print(f"Image spans: {image_spans}")
#         print(f"Instruction span: {inst_span}")
#         print(f"Desc span: {desc_span}")

#         # Sanity check each image span is all image tokens
#         image_token_id = vis_processor.image_token_id
#         for i, (s, e) in enumerate(image_spans):
#             assert all(t == image_token_id for t in seq[s:e]), f"image span {i} mismatch"

#         inst_dec = tokenizer.decode(seq[inst_span[0]:inst_span[1]], skip_special_tokens=False)
#         desc_dec = tokenizer.decode(seq[desc_span[0]:desc_span[1]], skip_special_tokens=False)

#         print(f"Sample Instruction: {sample_inst}")
#         print("inst slice preview:", inst_dec[:200])
#         print(f"Sample Description: {sample_desc}")
#         print("desc slice preview:", desc_dec[:200])

#         print("Exiting")
#         sys.exit()

#         # TODO: replace with your actual return format once debugging is done
#         # return target_parts, labels_per_sample

#     # ---------------- main loop (kept as close to your structure as possible) ----------------
#     if use_augmentation:
#         images_of_batch = batch["images"]
#         aug_images_of_batch = batch["aug_images"]
#         insts_of_batch = batch["inst"]
#         descs_of_batch = batch["desc"]

#         for sample_images, sample_aug_image, sample_inst, sample_desc in zip(
#             images_of_batch, aug_images_of_batch, insts_of_batch, descs_of_batch
#         ):
#             conversation = [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "text_part": "image1", "data": None},
#                     {"type": "text", "text_part": "inst", "text": "<INST>" + sample_inst + "</INST>"},
#                     {"type": "text", "text_part": "desc", "text": "<DESC>" + sample_desc + "</DESC>"},
#                 ]
#             }]

#             # FIX: you were missing '+' between string literals
#             prompt = (
#                 "<IMG1>" + "<image>" + "</IMG1>\n"
#                 + "<IMG2>" + "<image>" + "</IMG2>\n"
#                 + "<INST>" + sample_inst + "</INST>\n"
#                 + "<DESC>" + sample_desc + "</DESC>"
#             )

#             # If sample_images is a single image, pass it directly; if it's a list, pass the list.
#             images_arg = [sample_images, sample_images]
#             inputs = vis_processor(images=images_arg, text=prompt, return_tensors="pt")
#             input_ids = inputs["input_ids"][0]

#             print("Printing input keys")
#             print(inputs.keys())
#             print(f"Input_ids shape: {np.array(inputs['input_ids']).shape}")

#             conversation[0]["content"][0]["data"] = sample_images  # temp

#             inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
#             if "pixel_values" in inputs:
#                 inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

#             with torch.inference_mode():
#                 out = model(**inputs, return_dict=True)
#                 logits = out.logits

#             print(f"Logits shape: {logits.size()}")

#             target_parts, labels_per_sample = _get_parts_from_one_sample(
#                 conversation, prompt, input_ids, inputs, logits,
#                 tokenizer, vis_processor, parts, sample_inst, sample_desc
#             )


































def mod_infer_batch_hulu(model, batch, tokenizer, vis_processor, parts, use_augmentation):
    def _get_parts_from_one_sample(conversation, prompt, input_ids, inputs, logits, tokenizer, vis_processor, parts, sample_inst, sample_desc):
        def visual_token_count_from_inputs(inputs, image_index=0) -> int:
            gs = inputs["grid_sizes"][image_index]     # tensor like [T, H, W] or [1, H, W]
            ms = inputs["merge_sizes"][image_index]    # int or tensor scalar

            if hasattr(ms, "item"):
                ms = int(ms.item())

            T, H, W = [int(x) for x in gs.tolist()]    # e.g. [1, 48, 48]
            # processor logic: if T==1 => one image; if T>1 => video/3d treated like multiple frames
            return (T if T > 1 else 1) * (H // ms) * (W // ms)
        
        def enc_len(tokenizer, s: str) -> int:
            return len(tokenizer.encode(s, add_special_tokens=False))
        
        def compute_spans(tokenizer, inputs, vis_processor, inst: str, desc: str):
            n_img = visual_token_count_from_inputs(inputs, 0)

            before_img = "<IMG1>"
            after_img = "</IMG1>\n<INST>"
            between_inst_desc = "</INST>\n<DESC>"
            after_desc = "</DESC>"

            before_len     = enc_len(tokenizer, before_img)
            after_image_len= enc_len(tokenizer, after_img)
            inst_len       = enc_len(tokenizer, inst)
            between_len    = enc_len(tokenizer, between_inst_desc)
            desc_len       = enc_len(tokenizer, desc)
            after_desc_len = enc_len(tokenizer, after_desc)

            idx = 0

            # account for <IMG1>
            idx += before_len

            image_span = (idx, idx + n_img)
            idx += n_img

            idx += after_image_len

            inst_span = (idx, idx + inst_len)
            idx += inst_len

            idx += between_len

            desc_span = (idx, idx + desc_len)
            idx += desc_len

            idx += after_desc_len

            return image_span, inst_span, desc_span

        
        n_img = visual_token_count_from_inputs(inputs, 0)
        image_token_id = vis_processor.image_token_id
        
        
        seq = input_ids.tolist()
        image_span, inst_span, desc_span = compute_spans(tokenizer, inputs, vis_processor, sample_inst, sample_desc)
        # print(f'image_span: {image_span}')
        # print(f'inst_span: {inst_span}')
        # print(f'desc_span: {desc_span}')
        
        # print(f"Image span: {image_span}")
        # print(f"Instruction span: {inst_span}")
        # print(f"Desc span: {desc_span}")


        # # 1) Image region should be entirely <image> tokens
        # # assert all(t == image_token_id for t in seq[image_span[0]:image_span[1]]), "image span mismatch"

        # # 2) Decode inst/desc slices should match (mostly) your inst/desc
        # inst_dec = tokenizer.decode(seq[inst_span[0]:inst_span[1]], skip_special_tokens=False)
        # desc_dec = tokenizer.decode(seq[desc_span[0]:desc_span[1]], skip_special_tokens=False)
        # print(f"Sample Instruction: {sample_inst}")
        # print("inst slice preview:", inst_dec)
        # print(f"Sample Description: {sample_desc}")
        # print("desc slice preview:", desc_dec)

                            
        # print("Exiting")
        # sys.exit()

        # standardize tensors
        # input_ids is [seq] (because you pass inputs["input_ids"][0])
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=logits.device)
        else:
            input_ids = input_ids.to(logits.device)

        # logits is [1, seq, vocab] from Hulu
        if logits.dim() == 3:
            logits_2d = logits[0]   # [seq, vocab]
        else:
            logits_2d = logits      # assume already [seq, vocab]

        seq_len = input_ids.numel()
        assert logits_2d.size(0) == seq_len, f"seq mismatch: input_ids={seq_len}, logits={logits_2d.size(0)}"

        img_s, img_e = image_span
        inst_s, inst_e = inst_span
        desc_s, desc_e = desc_span

        # ---------------- build target_parts like original ----------------
        target_parts = {}
        for p in parts:
            if p not in target_parts:
                target_parts[p] = {"input_ids": [], "probabilities": [], "log_probabilities": []}

            if p == "img":
                _slice = slice(img_s, img_e)
            elif p == "inst_desp":
                _slice = slice(inst_s, desc_e)   # inst + (markers between) + desc? (if you want ONLY bodies, change)
            elif p == "inst":
                _slice = slice(inst_s, inst_e)
            elif p in ["desp", "desc"]:
                _slice = slice(desc_s, desc_e)
            else:
                raise ValueError(f"Not supported goal {p}")

            # input_ids for that part
            target_parts[p]["input_ids"].append(input_ids[_slice])

            # logits for that part
            part_logits = logits_2d[_slice, :]  # [part_len, vocab]
            target_parts[p]["probabilities"].append(torch.nn.functional.softmax(part_logits, dim=-1))
            target_parts[p]["log_probabilities"].append(torch.nn.functional.log_softmax(part_logits, dim=-1))

        # ---------------- labels_per_sample like original ----------------
        labels = [''] * seq_len
        labels[img_s:img_e]   = ['img']  * (img_e - img_s)
        labels[inst_s:inst_e] = ['inst'] * (inst_e - inst_s)
        labels[desc_s:desc_e] = ['desc'] * (desc_e - desc_s)

        return target_parts, [labels]

            
        
    
    if use_augmentation:
        # print('use_aug true')
        images_of_batch = batch["images"]
        aug_images_of_batch = batch["aug_images"]
        insts_of_batch = batch["inst"]
        descs_of_batch = batch["desc"]
        
        total_parts = {
        "orig":  list()   
        }

        total_token_labels = list()
        
        

        
        for sample_images, sample_aug_image, sample_inst, sample_desc in zip(images_of_batch, aug_images_of_batch, insts_of_batch, descs_of_batch):
            # Need to change to accept other modes not just image
            conversation = list()
            
            # for img in sample_images
            
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "text_part": "image1", "data": None},
                    {"type": "text", "text_part": "inst", "text": "<INST>" + sample_inst + "</INST>"},
                    {"type": "text", "text_part": "desc", "text": "<DESC>" + sample_desc + "</DESC>"},
                ]
            }]
            

            
            # Need to change so it accepts multiple images
            # Format of encoder special tokens
            prompt = (
                "<IMG1>" + "<image>" + "</IMG1>\n"
                "<INST>" + sample_inst + "</INST>\n"
                "<DESC>" + sample_desc + "</DESC>"
            )

            inputs = vis_processor(images=[sample_images], text=prompt, return_tensors="pt")
            
            # def n_img_from_inputs(inputs):
            #     gs = inputs["grid_sizes"][0]
            #     ms = inputs["merge_sizes"][0]
            #     ms = int(ms.item()) if hasattr(ms, "item") else int(ms)
            #     T, H, W = [int(x) for x in gs.tolist()]
            #     return (T if T > 1 else 1) * (H // ms) * (W // ms), (T, H, W), ms

            # n_img, (T,H,W), ms = n_img_from_inputs(inputs)
            # print('ORIGINAL')
            # print("PIL size:", (sample_images.size if hasattr(sample_images, "size") else type(sample_images)))
            # print("grid_sizes:", (T,H,W), "merge:", ms, "n_img:", n_img)
            
            input_ids = inputs["input_ids"][0]
            
            # print("Printing input keys")
            # print(inputs.keys())
            
            # print(f"Input_ids shape: {np.array(inputs['input_ids']).shape}")
            
            # Temp fix, not good
            conversation[0]['content'][0]['data'] = sample_images
            
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
            with torch.inference_mode():
                out = model(**inputs, return_dict=True)
                logits = out.logits
            
            
            # print("Exiting")
            # sys.exit()
                
            # print('for orig')
            target_parts, labels_per_sample = _get_parts_from_one_sample(conversation, prompt, input_ids, inputs, logits, tokenizer, vis_processor, parts, sample_inst, sample_desc)
                
            if len(total_parts["orig"]) == 0:
                total_parts["orig"].append(target_parts)
            else:
                for _part in parts:
                    for _key in total_parts["orig"][0][_part].keys():
                        total_parts["orig"][0][_part][_key].extend(target_parts[_part][_key])

            
            
            # 2. Conduct inference using the augmented images
            for k, aug_images in sample_aug_image.items():
                    # For each augmentation type

                    if k not in total_parts:
                        # Initialize the list for saving each setting
                        total_parts[k] = [None for _ in range(len(aug_images))]

                    for _setting_idx, _aug_img in enumerate(aug_images):
                        
                        
                        conversation = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "text_part": "image1", "data": None},
                                {"type": "text", "text_part": "inst", "text": "<INST>" + sample_inst + "</INST>"},
                                {"type": "text", "text_part": "desc", "text": "<DESC>" + sample_desc + "</DESC>"},
                            ]
                        }]
                        

                        
                        # Need to change so it accepts multiple images
                        # Format of encoder special tokens
                        prompt = (
                            "<IMG1>" + "<image>" + "</IMG1>\n"
                            "<INST>" + sample_inst + "</INST>\n"
                            "<DESC>" + sample_desc + "</DESC>"
                        )

                        inputs = vis_processor(images=[_aug_img], text=prompt, return_tensors="pt")
                        # n_img, (T,H,W), ms = n_img_from_inputs(inputs)
                        # print('AUG')
                        # print("PIL size:", (_aug_img.size if hasattr(_aug_img, "size") else type(_aug_img)))
                        # print("grid_sizes:", (T,H,W), "merge:", ms, "n_img:", n_img)
                        
                        # sys.exit()
                        
                        input_ids = inputs["input_ids"][0]
                        
                        # print("Printing input keys")
                        # print(inputs.keys())
                        
                        # print(f"Input_ids shape: {np.array(inputs['input_ids']).shape}")
                        
                        # Temp fix, not good
                        conversation[0]['content'][0]['data'] = _aug_img
                        
                        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                        if "pixel_values" in inputs:
                            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                            
                        with torch.inference_mode():
                            out = model(**inputs, return_dict=True)
                            logits = out.logits
                            
                        
                        
                        # print("Exiting")
                        # sys.exit()
                        # print(f"for aug: {k}")
                        target_parts, labels_per_sample = _get_parts_from_one_sample(conversation, prompt, input_ids, inputs, logits, tokenizer, vis_processor, parts, sample_inst, sample_desc)
                        
                        
                        if total_parts[k][_setting_idx] == None:
                            total_parts[k][_setting_idx] = target_parts
                        else:
                            for _part in parts:
                                # Insert (input_ids, probabilities, log_probabilities) from each part to the corresponding setting
                                for _key in total_parts[k][_setting_idx][_part].keys():
                                    total_parts[k][_setting_idx][_part][_key].extend(target_parts[_part][_key]) 
            # print('exiting')
            # sys.exit()
                                    
        # print(f"total_parts['org'] length: {len(total_parts['orig'])}")

        return total_parts, total_token_labels    
    
    else:
        raise NotImplementedError("Use Augmentation is False, not implimented yet")
        return None
