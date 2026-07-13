from src.model.generate import generate, generate_a_batch
from src.model.infer import mod_infer_batch, mod_infer_batch_minigpt, mod_infer_batch_hulu
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from src.misc import load_conversation_template
from minigpt.minigpt4.common.registry import registry
from minigpt.minigpt4.common.config import Config
from minigpt.minigpt4.conversation.interact import Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2, CONV_VISION_minigptv2
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch

def load_target_model(cfg):
    """
    cfg: target model config
    """
    if cfg.target_model.type == "llava":
        model_name = get_model_name_from_path(cfg.target_model.model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            cfg.target_model.model_path, 
            cfg.target_model.model_base, 
            model_name, 
            gpu_id=cfg.target_model.gpu_id,
            cache_dir=cfg.path.cache_dir
        )
        conv_mode = load_conversation_template(model_name)

        return model, tokenizer, image_processor, conv_mode

    elif cfg.target_model.type == "minigpt":

        _config = Config(cfg.target_model)

        model_config = _config.model_cfg
        model_config.device_8bit = cfg.target_model.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(cfg.target_model.gpu_id))

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2,
                     'minigptv2': CONV_VISION_minigptv2}
        CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = _config.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name).from_config(vis_processor_cfg)

        return model, vis_processor, CONV_VISION
    
    elif cfg.target_model.type == "hulu_med":

        # Load the original full Hulu-Med base model.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        # Optionally apply your fine-tuned LoRA adapter.
        adapter_path = cfg.target_model.get("adapter_path", None)

        if adapter_path is not None:
            print(f"Loading Hulu-Med LoRA adapter from: {adapter_path}")

            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                is_trainable=False,
            )

        model.eval()

        # Keep the processor tied to the original base model.
        hulu_processor = AutoProcessor.from_pretrained(
            cfg.target_model.model_path,
            trust_remote_code=True,
        )

        tokenizer = hulu_processor.tokenizer

        return model, tokenizer, hulu_processor


    else:
        raise ValueError(f"Unable to recognize the model {cfg.type}")        