import torch
from tqdm import tqdm
from minigpt.minigpt4.conversation.interact import Interact
from pathlib import Path

def generate_all_response(model, tokenizer, dataloader, num_gen_tokens):
    outputs = list()
    for _idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensor"].cuda()
        image_sizes = batch["image_size"].cuda()

        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cache=True
        )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for _out_text in output_text:
            outputs.append(_out_text.strip())

    return outputs

def generate_a_batch(model, tokenizer, batch, num_gen_tokens, use_augmentation):
    if use_augmentation:
        outputs = dict()
        input_ids = batch["input_ids"]
        image_sizes = batch["image_sizes"].cuda()

        # 1.Generate using the original images
        orig_image_tensors = batch["orig_image_tensors"].cuda()

        output_ids = model.generate(
            input_ids,
            images=orig_image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cache=True
        )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs["orig"] = list()
        for _out_text in output_text: 
            outputs["orig"].append(_out_text)

        # 2. Generate using each augmented view
        for k, aug_imgs in batch["aug_image_tensors"].items():
            outputs[k] = list()
            for _aug_img in aug_imgs:
                output_ids = model.generate(
                    input_ids,
                    images=_aug_img.cuda(),
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=num_gen_tokens,
                    use_cache=True
                )
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for _out_text in output_text: 
                    outputs[k].append(_out_text)        

    else:
        outputs = list()
        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensors"].cuda()
        image_sizes = batch["image_sizes"].cuda()

        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cache=True
        )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        for _out_text in output_text:
            outputs.append(_out_text.strip())
        
    return outputs

def generate_a_batch_minigpt(model, vis_encoder, batch, num_gen_tokens, chat_state, gpu_id, use_augmentation):
    """
    Generation function from mini-gpt-4
    It is actually not batched: each image is processed independently.
    Args:
        model: model,

    """
    outputs = list()
    if use_augmentation:
        raise NotImplementedError()
    else:
        images = batch["images"]
        texts = batch["texts"]
        chat = Interact(model, vis_encoder, device="cuda:{}".format(gpu_id))
        for _img, _text in zip(images, texts):
            _img_list = []
            _chat_state = chat_state.copy()
            llm_message = chat.upload_img(_img[0], _chat_state, _img_list)
            chat.encode_img(_img_list)
            chat.ask(_text, _chat_state)
        
            gen_ = chat.get_generate_output(conv=_chat_state,
                                            img_list=_img_list,
                                            max_new_tokens=num_gen_tokens,
                                            do_sample=False)
            output_text = chat.model.llama_tokenizer.decode(gen_[0], skip_special_tokens=True)
            outputs.append(output_text)

    return outputs

def generate_a_batch_med_hulu(model, img_processor, batch, num_gen_tokens, use_augmentation):
    """
    Generation function from med_hulu
    It is actually not batched: each image set is processed independently.
    Args:
        model: model,

    """
    outputs = list()
    if use_augmentation:
        raise NotImplementedError()
    else:
        conversations = batch["conversations"]
        
        for conversation in conversations:
            
            inputs = img_processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            output_ids = model.generate(**inputs, max_new_tokens=num_gen_tokens)
            outputs_no_think = img_processor.batch_decode(
                output_ids, 
                skip_special_tokens=True,
                use_think=False  
            )[0].strip()
            
            outputs.append(outputs_no_think)
            
    return outputs


class BatchProcessor:
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
        indices: 1D vector has index of samples in the batch
        input_ids : 2D vector [sample_idx, input_ids]
        image_tensors: 4D vector [sample_idx, channel_dim, width, height]
        image_sizes: 3D vector [sample_idx, height, width]
        """
        if self.current_batch == self.num_batch:
            raise StopIteration
 
        indices = list()
        input_ids = list()
        image_tensors = list()
        image_sizes = list()
        
        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size        

        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            image_tensors.append(self.dataset[_idx]["image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])

        self.current_batch+=1

        return {
            "indices" : torch.tensor(indices),
            "input_ids" : torch.tensor(input_ids),
            "image_tensors" : torch.tensor(image_tensors, dtype=torch.float16),
            "image_sizes" : torch.tensor(image_sizes)
        }

    def _get_augmented_batch(self):
        indices = list()
        input_ids = list()
        orig_image_tensors = list()
        image_sizes = list()
        aug_image_tensors = dict()

        batch_begin = self.current_batch * self.batch_size
        if self.current_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size        

        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            orig_image_tensors.append(self.dataset[_idx]["image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])

            for k, aug_imgs in self.dataset[_idx]["aug_img_tensors"].items():
                if k not in aug_image_tensors :
                    aug_image_tensors[k] = [[] for _ in range(len(aug_imgs))]
                for _aug_idx, _aug_img in enumerate(aug_imgs):
                    aug_image_tensors[k][_aug_idx].append(_aug_img)

        for k, aug_imgs in aug_image_tensors.items():
            for _aug_idx in range(len(aug_imgs)):
                aug_image_tensors[k][_aug_idx] = torch.tensor(aug_image_tensors[k][_aug_idx])

        self.current_batch+=1
        return {
            "indices" : indices,
            "input_ids": torch.tensor(input_ids),
            "orig_image_tensors": torch.tensor(orig_image_tensors, dtype=torch.float16),
            "aug_image_tensors": aug_image_tensors
        }

class BatchProcessor_minigpt:
    def __init__(self, dataset, batch_size, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % batch_size == 0 else int(len(dataset)/self.batch_size)+1

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_augmentation:
            raise NotImplementedError()
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

        batch_imgs = list()
        batch_texts = list()
        batch_indices = list()
        for _idx in range(batch_begin, batch_end):
            batch_imgs.append(self.dataset[_idx]["images"])
            batch_texts.append(self.dataset[_idx]["texts"])
            batch_indices.append(self.dataset[_idx]["indices"])

        self.current_batch+=1
        
        return {
            "indices": batch_indices,
            "images": batch_imgs,
            "texts": batch_texts
        }
        
class BatchProcessor_hulu_med:
    def __init__(self, dataset, batch_size, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size) if len(dataset) % batch_size == 0 else int(len(dataset)/self.batch_size)+1

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_augmentation:
            raise NotImplementedError()
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

        batch_converstaions = list()
        batch_indices = list()
        for _idx in range(batch_begin, batch_end):
            conversation = [
                {
                "role": "user",
                "content": []
                }
            ]
            
            if self.dataset[_idx]["image_paths"] is None:
                print(f'None found in batch {self.current_batch}, index {_idx}  ')
                print('self.dataset[_idx]["image_paths"]: ')
                print(f'{self.dataset[_idx]["image_paths"]}')
            
            for image_path in self.dataset[_idx]["image_paths"]:
                conversation[0]['content'].append(
                    {
                        "type":"image",
                        "image": {
                            "image_path": image_path,
                        }
                    }
                )
            
            conversation[0]['content'].append(
                {
                    "type": "text", 
                    "text": str(self.dataset[_idx]["texts"])
                }
            )

            
            
            batch_indices.append(self.dataset[_idx]["indices"])
            batch_converstaions.append(conversation)

        self.current_batch+=1
        
        return {
            "indices": batch_indices,
            "conversations": batch_converstaions
        }

def generate(model, dataset, cfg, tokenizer=None, vis_encoder=None, chat_state=None, gpu_id=None):
    indices = list()
    outputs = list()
    if cfg.target_model.type == "llava":
        assert tokenizer != None
        num_gen_tokens = cfg.generation.num_gen_tokens
        batch_generator = BatchProcessor(dataset=dataset,
                                        batch_size=cfg.generation.batch_size,
                                        use_augmentation=cfg.generation.use_augmentation)
        
        # Calculate total number of batches for progress bar
        total_batches = len(batch_generator)
        
        # Add progress bar
        for b_idx, batch in enumerate(tqdm(batch_generator,
                                total=total_batches, 
                                desc="Generating responses", 
                                unit="batch")):
            _outputs = generate_a_batch(model, tokenizer, batch, cfg.generation.num_gen_tokens, cfg.generation.use_augmentation)

            outputs.extend(_outputs)
            indices.extend(batch["indices"].tolist())
    
    elif cfg.target_model.type == "minigpt":
        assert vis_encoder != None
        assert chat_state != None
        assert gpu_id != None

        num_gen_tokens = cfg.generation.num_gen_tokens
        batch_generator = BatchProcessor_minigpt(dataset=dataset,
                                                 batch_size=cfg.generation.batch_size,
                                                 use_augmentation=cfg.generation.use_augmentation)
        total_batches = len(batch_generator)

        for b_idx, batch in enumerate(tqdm(batch_generator,
                                total=total_batches,
                                desc="Generating responses",
                                unit="batch")):
            _outputs = generate_a_batch_minigpt(model, vis_encoder, batch, num_gen_tokens,
                                                chat_state, gpu_id, cfg.generation.use_augmentation)

            outputs.extend(_outputs)
            indices.extend(batch["indices"])
            
    elif cfg.target_model.type == "hulu_med":
        assert tokenizer != None
        assert vis_encoder != None
        
        num_gen_tokens = cfg.generation.num_gen_tokens
        
        batch_generator = BatchProcessor_hulu_med(dataset=dataset,
                                                 batch_size=cfg.generation.batch_size,
                                                 use_augmentation=cfg.generation.use_augmentation)
        total_batches = len(batch_generator)
        
        for b_idx, batch in enumerate(tqdm(batch_generator,
                                total=total_batches,
                                desc="Generating responses",
                                unit="batch")):
            _outputs = generate_a_batch_med_hulu(model, vis_encoder, batch, num_gen_tokens,
                                                 cfg.generation.use_augmentation)

            outputs.extend(_outputs)
            indices.extend(batch["indices"])
        
        

    return indices, outputs
