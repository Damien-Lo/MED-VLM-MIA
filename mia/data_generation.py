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


sys.path.append("./minigpt")

"""
A debug code for data generation

"""
@hydra.main(version_base=None, config_path="./config", config_name="run_img")
def main(cfg):

    print("Full Config Paramters:")
    print(cfg)
    print("\n")
    print("AUGMENTATIONS USED:")
    print(cfg.data.augmentations)
    print("\nREQUESTED DATA")        
        
    print('''
          \n \n
          ==================================================
                            LOADING MODEL
          ==================================================
          '''
          )

    # # Load the target model
    target_model = load_target_model(cfg)
    print("Model Loaded")

    # Do the data generation
    text = cfg.prompt.text
    if cfg.target_model.type == "llava":
        model, tokenizer, image_processor, conv_mode = target_model
        gen_data = get_generation_data(cfg, cfg.target_model.type, text, tokenizer, image_processor, model.config, conv_mode)
        # idxs, descriptions = generate(model, tokenizer, gen_data, cfg)
        idxs, descriptions = generate(model, gen_data, cfg, tokenizer)

    elif cfg.target_model.type == "minigpt":
        model, vis_encoder, chat_state = target_model
        gpu_id = model.device.index if hasattr(model, "device") and hasattr(model.device, "index") else 0
        gen_data = get_generation_data(cfg, cfg.target_model.type, text)
        idxs, descriptions = generate(model, gen_data, cfg,
                                        vis_encoder=vis_encoder,
                                        chat_state=chat_state,
                                        gpu_id=gpu_id)
        
    elif cfg.target_model.type == "hulu_med":
        model, tokenizer, image_processor = load_target_model(cfg)
        gen_data = get_generation_data(cfg, cfg.target_model.type, text)
    
        idxs, descriptions = generate(model, gen_data, cfg, tokenizer, image_processor)

        
    else:
        raise ValueError(f"Unexpected model type {cfg.target_model.type}")

    # Save the generated text
    # save_path = os.path.join(cfg.path.output_dir, "generation", str(cfg.target_model.type), str(cfg.data.subset))
    # save_path = cfg.path.output_dir
    # os.makedirs(save_path, exist_ok=True)

    sentences = {
        "idxs": idxs,
        "sentences": descriptions
    }

    import json
    with open(cfg.path.output_dir, 'w') as f:
        json.dump(sentences, f, indent=2)

if __name__ == "__main__":
    main()
