import torch

def get_parts_slices(prompt_0, prompt_1, desc_shape):
    
    img_slice = slice(len(prompt_0),-len(prompt_1)+1)
    inst_desc = slice(-len(prompt_1)+1, None)
    inst = slice(-len(prompt_1)+1,-desc_shape)
    desc = slice(-desc_shape, None)

    img_loss_slice = slice(img_slice.start-1, img_slice.stop-1)

    return img_loss_slice, img_slice, inst_desc, inst, desc

