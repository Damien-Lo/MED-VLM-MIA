import re
import torch
import requests
from PIL import Image
from io import BytesIO
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets, load_from_disk
from src.data.augmentations import get_augmentations
from torchvision import transforms
import numpy as np
import sys
from datasets.features import Image as HFImage

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.conversation import conv_templates


# Global variables that need to be defined or imported

def load_image(image_file):
    """
    Opens an image in a image_file,
    returns PIL.Image instance
    """
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_generation_data(cfg, model_type, _dataset, text, tokenizer=None, image_processor=None, model_config=None, conv_mode=None):
    """
    cfg :  dataset config
    tokenizer: tokenizer instance
    text: Input instruction text (Describe this image)
    model_config: model.config
    conv: conv_mode
    model_tyep: cfg.target_model.type: Either llava or minigpt
    """
    
    # if cfg.data.dataset == 'JaineLi/VL-MIA-image' and cfg.data.subset in ['img_Flickr', 'img_dalle']:
    #     _dataset = load_dataset(path=cfg.data.dataset,
    #                         name=cfg.data.subset,
    #                         split=cfg.data.split,
    #                         cache_dir=cfg.path.cache_dir)
    # else:
    #     _dataset = Dataset.from_file(cfg.data.subset)
    
    # _dataset = _dataset.add_column("indices", list(range(len(_dataset))))
    if model_type == "llava":
        if cfg.generation.use_augmentation:
            _dataset = _dataset.map(convert_to_aug_generation_input_ids,
                                batched=True,
                                load_from_cache_file=False,
                                fn_kwargs={
                                    "tokenizer": tokenizer,
                                    "image_processor": image_processor,
                                    "instruction": text,
                                    "model_config": model_config,
                                    "conv_mode": conv_mode,
                                    "cfg": cfg
                                })
            
        else:
            _dataset = _dataset.map(convert_to_generation_input_ids,
                                batched=True,
                                load_from_cache_file=False,
                                fn_kwargs={
                                    "tokenizer": tokenizer,
                                    "image_processor": image_processor,
                                    "instruction": text,
                                    "model_config": model_config,
                                    "conv_mode": conv_mode,
                                    "cfg": cfg
                                })
    elif model_type == "minigpt":
        if cfg.generation.use_augmentation:
            raise NotImplementedError()
        else:
            _dataset = _dataset.map(convert_to_generation_raw,
                                batched=True,
                                load_from_cache_file=False,
                                fn_kwargs={
                                    "instruction": text
                                })
            
    elif model_type == "hulu_med":
        if cfg.generation.use_augmentation:
            raise NotImplementedError()
        else:    
            _dataset = _dataset.map(convert_to_generation_raw_hulu,
                                batched=True,
                                load_from_cache_file=False,
                                fn_kwargs={
                                    "instruction": text
                                })

    return _dataset


def get_mod_infer_data(cfg, text, _dataset, model_config=None, tokenizer=None, image_processor=None, conv_mode=None):
    """
    cfg :  dataset config
    descriptions: generated responses
    tokenizer: tokenizer instance
    text: Input instruction text (Describe this image)
    model_config: model.config
    conv: conv from cfg.target_model
    """
    
    true_class_labels = _dataset["label"]
    image_sampled_indicies = []
    categorised_image_sampled_indicies = dict()
    
    if cfg.img_metrics.get_raw_images > 0:
        image_sampled_indicies = np.sort(np.concatenate([
        np.where(np.array(true_class_labels) == 1)[0][:cfg.img_metrics.get_raw_images],
        np.where(np.array(true_class_labels) == 0)[0][:cfg.img_metrics.get_raw_images]
        ]))
        
        categorised_image_sampled_indicies = {'members':np.where(np.array(true_class_labels) == 1)[0][:cfg.img_metrics.get_raw_images].tolist(),
                    'non_members':np.where(np.array(true_class_labels) == 0)[0][:cfg.img_metrics.get_raw_images].tolist()}
        
        print(f"Raw Image Indecies: {image_sampled_indicies}")
    
    if cfg.target_model.type == "llava":
        if cfg.inference.use_augmentation:
            _dataset = _dataset.map(convert_to_augmentation_mod_infer,
                                batched=True,
                                load_from_cache_file=False,
                                keep_in_memory=True,
                                fn_kwargs={
                                    "tokenizer": tokenizer,
                                    "image_processor": image_processor,
                                    "instruction": text,
                                    "model_config": model_config,
                                    "conv_mode": conv_mode,
                                    "cfg": cfg,
                                    "image_sampled_indicies": image_sampled_indicies
                                })

        else:
            _dataset = _dataset.map(convert_to_mod_infer,
                                batched=True,
                                load_from_cache_file=False,
                                keep_in_memory=True,
                                fn_kwargs={
                                    "tokenizer": tokenizer,
                                    "image_processor": image_processor,
                                    "instruction": text,
                                    "model_config": model_config,
                                    "conv_mode": conv_mode,
                                    "cfg": cfg
                                })

    elif cfg.target_model.type == "minigpt":
        if cfg.inference.use_augmentation:
            print("Getting convert_to_augmentation_mod_infer_minigpt")
            _dataset = _dataset.map(convert_to_augmentation_mod_infer_minigpt,
                                    batched=True,
                                    load_from_cache_file=False,
                                    keep_in_memory=True,
                                    fn_kwargs={
                                        "instruction": text,
                                        "cfg": cfg,
                                        "image_sampled_indicies": image_sampled_indicies
                                    })
            print("Loaded convert_to_augmentation_mod_infer_minigpt")
        else:
            print("Getting convert_to_mod_infer_minigpt")
            _dataset = _dataset.map(convert_to_mod_infer_minigpt,
                                    batched=True,
                                    load_from_cache_file=False,
                                    keep_in_memory=True,)
            print("Loaded convert_to_mod_infer_minigpt")
            
    elif cfg.target_model.type == "hulu_med":
        if cfg.inference.use_augmentation:
            print("Getting convert_to_augmentation_mod_infer_hulu")
            _dataset = _dataset.map(convert_to_augmentation_mod_infer_hulu,
                                    batched=True,
                                    load_from_cache_file=False,
                                    keep_in_memory=True,
                                    fn_kwargs={
                                        "instruction": text,
                                        "cfg": cfg,
                                        "image_sampled_indicies": image_sampled_indicies
                                    })
            print("Loaded convert_to_augmentation_mod_infer_hulu")
        else:
            print("Getting convert_to_mod_infer_hulu")
            _dataset = _dataset.map(convert_to_mod_infer_hulu,
                                    batched=True,
                                    load_from_cache_file=False,
                                    keep_in_memory=True,)
            print("Loaded convert_to_mod_infer_hulu")
        
    else:
        raise ValueError(f"Unknown model type {cfg.target_model.type}")

    return _dataset, categorised_image_sampled_indicies


def convert_to_generation_input_ids(examples, tokenizer, image_processor, instruction, model_config, conv_mode, cfg):
    """
    Preprocess the generation dataset
    """
    image_paths = examples["image"]
    indices = examples["indices"]
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = instruction
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    # Load The conversation with the query
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    all_indices = list()
    all_input_ids = list()
    all_image_tensors = list()
    all_image_sizes = list()
    for _image_path, _indices in zip(image_paths, indices):
        images = load_images([_image_path])
        image_sizes = [x.size for x in images]
        image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        all_indices.append(_indices)
        all_input_ids.append(input_ids)
        all_image_tensors.append(image_tensor.squeeze(0))
        all_image_sizes.append(image_sizes[0])

    return {
        "indices": all_indices,
        "input_ids": all_input_ids,
        "image_tensors": all_image_tensors,
        "image_sizes": all_image_sizes
    }

def convert_to_aug_generation_input_ids(examples, tokenizer, image_processor, instruction, model_config, conv_mode, cfg):
    """
    Preprocess the generation dataset
    """
    indices = examples["indices"]
    image_paths = examples["image"]
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = instruction
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    # Load The conversation with the query
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    all_indices = list()
    all_input_ids = list()
    all_orig_image_tensors = list()
    all_image_sizes = list()
    all_aug_images = list()
    aug_dict = get_augmentations(cfg)
    for _image_path, _indices in zip(image_paths, indices):
        image = load_image(_image_path)  # Loading just one image
        orig_image_tensor = process_images(
            [image],
            image_processor,
            model_config
        )
        # Get augmented views
        image_sizes = image.size
        aug_imgs = dict()
        for k, aug_f_list in aug_dict.items():          # For Aug Type, Aug_Setting_List
            _aug_tensor_list = list()
            for _aug_f in aug_f_list:                   # For Setting In Aug_setting_List
                _aug_image = _aug_f(image)
                _aug_image_tensor = process_images(
                    [_aug_image],
                    image_processor,
                    model_config
                )
                _aug_tensor_list.append(_aug_image_tensor)
            aug_imgs[k] = _aug_tensor_list
        
        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        all_indices.append(_indices)
        all_aug_images.append(aug_imgs)
        all_input_ids.append(input_ids)
        all_orig_image_tensors.append(orig_image_tensor.squeeze(0))
        all_image_sizes.append(image_sizes)

    return {
        "indices": all_indices,
        "input_ids": all_input_ids,
        "orig_image_tensors": all_orig_image_tensors,
        "image_sizes": all_image_sizes,
        "aug_image_tensors": all_aug_images
    }

def convert_to_generation_raw(examples, instruction):
    image_paths = examples["image"]
    all_images = list()
    all_texts = list()
    for _image_path in image_paths:
        images = load_images([_image_path])
        all_images.append(images)
        all_texts.append(instruction)

    return {
        "indices": examples["indices"],
        "images": all_images,
        "texts": all_texts
    }
    
def convert_to_generation_raw_hulu(examples, instruction):
    # Not working as dataset has no column 'images' defined in mia.py line 169
    image_paths = examples["image"]
    all_image_paths = list()
    all_images = list()
    all_texts = list()
    for _image_path in image_paths:
        # images = load_images([_image_path])
        all_image_paths.append(_image_path)
        # all_images.append(images)
        all_texts.append(instruction)

    return {
        "indices": examples["indices"],
        # "images": all_images,
        # "image_paths": all_image_paths,
        "texts": all_texts
    }
    


## Mod-infer processors

def convert_to_mod_infer(examples, tokenizer, image_processor, instruction, model_config, conv_mode, cfg):
    """
    Preprocess the mod_infer dataset
    example: batched rows of dataset (has image paths and the descriptions paired with each image)
    instruction: instruction
    descriptions: generated descriptions
    """
    image_paths = examples["image"]
    descriptions = examples["desc"]
    qs = instruction
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    all_input_ids = list()
    all_image_tensors = list()
    all_image_sizes = list()
    all_img_slices = list()
    all_prompt_0 = list()
    all_prompt_1 = list()
    all_desc_shape = list()

    for _image_path, _description in zip(image_paths, descriptions):
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], _description)
        prompt = conv.get_prompt()[:-4]

        images = load_images([_image_path])
        image_sizes = [x.size for x in images]
        image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        all_input_ids.append(input_ids)
        all_image_tensors.append(image_tensor)
        all_image_sizes.append(image_sizes[0])

        desc_encoding = tokenizer(_description, return_tensors="pt", add_special_tokens = False).input_ids

        all_prompt_0.append(prompt_chunks[0])
        all_prompt_1.append(prompt_chunks[-1])
        all_desc_shape.append(desc_encoding.shape[1])

    return {
        "input_ids": all_input_ids,
        "image_tensors": all_image_tensors,
        "image_sizes" : all_image_sizes,
        "prompt_0" : all_prompt_0,
        "prompt_1" :  all_prompt_1,
        "desc_shape": all_desc_shape
    }

def convert_to_augmentation_mod_infer(examples, tokenizer, image_processor, instruction, model_config, conv_mode, cfg, image_sampled_indicies):
    
    """
    Preprocess the mod_infer dataset
    example: batched rows of dataset (has image paths and the descriptions paired with each image)
    instruction: instruction
    descriptions: generated descriptions
    """
    indices = examples["indices"]
    image_paths = examples["image"]
    descriptions = examples["desc"]
    qs = instruction
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    all_indices = list()
    all_input_ids = list()
    all_orig_image_tensors = list()
    all_orig_raw_images = list()
    all_aug_images = list()
    all_aug_raw_images = list()
    all_image_sizes = list()
    all_prompt_0 = list()
    all_prompt_1 = list()
    all_desc_shape = list()
    
    aug_dict = get_augmentations(cfg)

    for _indices, _image_path, _description in zip(indices, image_paths, descriptions):
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], _description)
        prompt = conv.get_prompt()[:-4]

        images = load_images([_image_path])
                
        image_sizes = [x.size for x in images]
        orig_image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        aug_imgs = dict()
        aug_raw_imgs = dict() # {aug_key: [aug_img, aug_img]}
        for k, aug_f_list in aug_dict.items():              # For Aug Type, Aug_Setting_List
            _aug_tensor_list = list()
            _aug_raw_image_list = list()
            for _aug_f in aug_f_list:                       # For Setting In Aug_setting_List
                _aug_image = _aug_f(images[0])
                _aug_image_tensor = process_images(
                    [_aug_image],
                    image_processor,
                    model_config
                )
                _aug_tensor_list.append(_aug_image_tensor)
                _aug_raw_image_list.append(np.array(_aug_image))
            aug_imgs[k] = _aug_tensor_list
            aug_raw_imgs[k] = _aug_raw_image_list
            
            
        if _indices in image_sampled_indicies:
            all_orig_raw_images.append(np.array(images[0]))
            all_aug_raw_images.append(aug_raw_imgs)
        else:
            all_orig_raw_images.append(None)
            all_aug_raw_images.append(None)
            

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        all_indices.append(_indices)
        all_input_ids.append(input_ids)
        all_orig_image_tensors.append(orig_image_tensor)
        all_image_sizes.append(image_sizes[0])
        all_aug_images.append(aug_imgs)

        desc_encoding = tokenizer(_description, return_tensors="pt", add_special_tokens = False).input_ids

        all_prompt_0.append(prompt_chunks[0])
        all_prompt_1.append(prompt_chunks[-1])
        all_desc_shape.append(desc_encoding.shape[1])

    return {
        "input_ids": all_input_ids,
        "orig_image_tensors": all_orig_image_tensors,
        "orig_raw_images": all_orig_raw_images,
        "aug_image_tensors": all_aug_images,
        "aug_raw_images": all_aug_raw_images,
        "image_sizes" : all_image_sizes,
        "prompt_0": all_prompt_0,
        "prompt_1": all_prompt_1,
        "desc_shape": all_desc_shape
    }

def convert_to_mod_infer_minigpt(examples, instruction):
    image_paths = examples["image"]
    all_images = list()
    all_texts = list()
    all_descriptions = list()
    for _image_path in image_paths:
        images = load_images([_image_path])
        all_images.append(images)
        all_texts.append(instruction)

    return {
        "indices": examples["indices"],
        "raw_images": all_images,
        "inst": all_texts,
        "desc": examples["desc"]
    }

def convert_to_augmentation_mod_infer_minigpt(examples, instruction, cfg, image_sampled_indicies):
    image_paths = examples["image"]
    all_orig_images = list()
    all_aug_images = list()
    all_texts = list()
    

    aug_dict = get_augmentations(cfg)

    for _image_path, _desc in zip(examples["image"], examples["desc"]):
        images = load_images([_image_path])
        
        aug_imgs = dict()
        for k, aug_f_list in aug_dict.items():
            _aug_img_list = list()
            for _aug_f in aug_f_list:
                _aug_img = _aug_f(images[0])
                if isinstance(_aug_img, dict):
                    raise ValueError("dict")
                _aug_img_list.append(_aug_img)
            aug_imgs[k] = _aug_img_list
        all_orig_images.append(images[0])
        all_aug_images.append(aug_imgs)
        all_texts.append(instruction)

    return {
        "indices": examples["indices"],
        "orig_images": all_orig_images,
        "aug_images": all_aug_images,
        # "orig_raw_images": all_orig_images,
        # "aug_raw_images": all_aug_images,
        "inst": all_texts,
        "desc": examples["desc"]
    }
    
    
def convert_to_mod_infer_hulu(examples, instruction):
    image_paths = examples["image"]
    all_images = list()
    all_texts = list()
    all_descriptions = list()
    for _image_path in image_paths:
        images = load_images([_image_path])
        all_images.append(images)
        all_texts.append(instruction)

    return {
        "indices": examples["indices"],
        "raw_images": all_images,
        "inst": all_texts,
        "desc": examples["desc"]
    }

def convert_to_augmentation_mod_infer_hulu(examples, instruction, cfg, image_sampled_indicies):
    image_paths = examples["image"]
    all_orig_images = list()
    all_aug_images = list()
    all_texts = list()
    

    aug_dict = get_augmentations(cfg)

    for _image_path, _desc in zip(examples["image"], examples["desc"]):
        images = load_images([_image_path])
        
        aug_imgs = dict()
        for k, aug_f_list in aug_dict.items():
            _aug_img_list = list()
            for _aug_f in aug_f_list:
                _aug_img = _aug_f(images[0])
                if isinstance(_aug_img, dict):
                    raise ValueError("dict")
                _aug_img_list.append(_aug_img)
            aug_imgs[k] = _aug_img_list
        all_orig_images.append(images[0])
        all_aug_images.append(aug_imgs)
        all_texts.append(instruction)

    return {
        "indices": examples["indices"],
        "orig_images": all_orig_images,
        "aug_images": all_aug_images,
        "orig_raw_images": all_orig_images,
        "aug_raw_images": all_aug_images,
        "inst": all_texts,
        "desc": examples["desc"]
    }
