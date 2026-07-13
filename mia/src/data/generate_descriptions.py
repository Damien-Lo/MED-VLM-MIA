import os
import sys
import json
import hydra
import torch
# from src.data import get_generation_data
from src.data.load_img_data import get_generation_data
from src.model import generate
from src.model import load_target_model
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from src.misc import load_conversation_template


def generate_descriptions(cfg, target_model, _dataset, out_path=None):
    # target_model = load_target_model(cfg)
    text = cfg.prompt.text
    if cfg.target_model.type == "llava":
        model, tokenizer, image_processor, conv_mode = target_model
        gen_data = get_generation_data(cfg, cfg.target_model.type, _dataset, text, tokenizer, image_processor, model.config, conv_mode)
        # idxs, descriptions = generate(model, tokenizer, gen_data, cfg)
        idxs, descriptions = generate(model, gen_data, cfg, tokenizer)

    elif cfg.target_model.type == "minigpt":
        model, vis_encoder, chat_state = target_model
        gpu_id = model.device.index if hasattr(model, "device") and hasattr(model.device, "index") else 0
        gen_data = get_generation_data(cfg, cfg.target_model.type, _dataset, text)
        idxs, descriptions = generate(model, gen_data, cfg,
                                        vis_encoder=vis_encoder,
                                        chat_state=chat_state,
                                        gpu_id=gpu_id)
    
    elif cfg.target_model.type == "hulu_med":
        model, tokenizer, image_processor = target_model
        gen_data = get_generation_data(cfg, cfg.target_model.type, _dataset, text, tokenizer, image_processor, model.config)
        
        # idxs, descriptions = generate(model, tokenizer, gen_data, cfg)
        idxs, descriptions = generate(model, gen_data, cfg, tokenizer, image_processor)
    
    else:
        raise ValueError(f"Unexpected model type {cfg.target_model.type}")
    
    
    sentences = {
        "idxs": idxs,
        "sentences": descriptions
    }
    
    if out_path != None:
        with open(out_path, 'w') as f:
            json.dump(sentences, f, indent=2)
            
    return sentences
        