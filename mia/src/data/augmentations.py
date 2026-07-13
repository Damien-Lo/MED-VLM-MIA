import torch
import torchvision.transforms as transforms
from functools import partial
from PIL import Image
import numpy as np

def get_augmentations(cfg):
    aug_dict = cfg.data.augmentations
    aug_func_dict = dict()
    aug_desc_dict = dict()
    for key, values in aug_dict.items():
        if key == "RandomResize" and values.use == True:
            _aug_f_list = list()
            _aug_desc_list = list()
            for _size, _scale, _ratio in zip(values.size, values.scale, values.ratio):
                aug_f = transforms.RandomResizedCrop(
                    size=_size,
                    scale=_scale,
                    ratio=_ratio
                )
                _aug_f_list.append(aug_f)
                _aug_desc_list.append({'size': _size, 'scale':_scale, 'ratio': _ratio})
                
            aug_func_dict[key] = _aug_f_list
            aug_desc_dict[key] = _aug_desc_list
        elif key == "RandomRotation" and values.use == True:
            _aug_f_list = list()
            _aug_desc_list = list()
            for _degrees in values.degrees:
                aug_f = transforms.RandomRotation(
                    degrees=_degrees
                )
                _aug_f_list.append(aug_f)
                _aug_desc_list.append({'degrees': _degrees})
            aug_func_dict[key] = _aug_f_list
            aug_desc_dict[key] = _aug_desc_list
        elif key == "RandomAffine" and values.use == True:
            _aug_f_list = list()
            _aug_desc_list = list()
            for _degree, _translate, _scale in zip(values.degrees, values.translate, values.scale):
                aug_f = transforms.RandomAffine(
                    degrees=_degree,
                    translate=_translate,
                    scale=_scale
                )
                _aug_f_list.append(aug_f)
                _aug_desc_list.append({'degrees': _degree, 'translate':_translate, 'scale': _scale})
            aug_func_dict[key] = _aug_f_list
            aug_desc_dict[key] = _aug_desc_list
        elif key == "ColorJitter" and values.use == True:
            _aug_f_list = list()
            _aug_desc_list = list()
            for _brightness, _contrast, _saturation, _hue in zip(values.brightness,
                                                                 values.contrast,
                                                                 values.saturation,
                                                                 values.hue):
                aug_f = transforms.ColorJitter(
                    brightness=_brightness,
                    contrast=_contrast,
                    saturation=_saturation,
                    hue=_hue
                )
                _aug_f_list.append(aug_f)
                _aug_desc_list.append({'brightness': _brightness, 'constrast': _contrast, 'saturation': _saturation, 'hue': _hue})
            aug_func_dict[key] = _aug_f_list
            aug_desc_dict[key] = _aug_desc_list
        elif key == "GaussianNoise" and values.use == True:
            _aug_f_list = list()
            _aug_desc_list = list()
            for _mean, _std in zip(values.mean, values.std):
                aug_f = AddGaussianNoisePIL(
                    mean=_mean,
                    std=_std,
                    clip=True
                )
                _aug_f_list.append(aug_f)
                _aug_desc_list.append({'mean': _mean, 'std': _std})
            aug_func_dict[key] = _aug_f_list
            aug_desc_dict[key] = _aug_desc_list
            
        elif key == "GaussianNoiseRipple" and values.use == True:
            _aug_f_list = list()
            _aug_desc_list = list()
            for _mean, _std, _start_patch in zip(values.mean, values.std, values.start_patch):
                aug_f = NoiseRippleAdderPIL(
                    mean=_mean,
                    std=_std,
                    clip=True,
                    start_patch = _start_patch
                )
                _aug_f_list.append(aug_f)
                _aug_desc_list.append({'mean': _mean, 'std': _std})
            aug_func_dict[key] = _aug_f_list
            aug_desc_dict[key] = _aug_desc_list
            
    return aug_func_dict, aug_desc_dict

# Custom Gaussian Noise transform for PIL images
class AddGaussianNoisePIL:
    def __init__(self, mean=0., std=10., clip=True):
        self.mean = mean
        self.std = std
        self.clip = clip

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        # Convert to NumPy array
        import numpy as np
        arr = np.array(image).astype(np.float32)

        # Add Gaussian noise
        noise = np.random.normal(self.mean, self.std, arr.shape)
        noisy = arr + noise

        # Clip values to valid range
        if self.clip:
            noisy = np.clip(noisy, 0, 255)

        # Convert back to PIL Image
        return Image.fromarray(noisy.astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, clip={self.clip})"
    
#TODO: Due to the global position and token awareness from the vision transfomer attention, the model might look at / place first priorty in looking at the middle tokens for example, 
# as such maybe adding noise to these would be better? But then you loose the 'inference' nature of judging how the model thinks about the actual content of the image, because you augmented it
# so the bet is that if you simply forget the attention and token awarenss, there is enough structure in the inherent patching positions (regardless of position context) that the noise you add at the top left 
# and so on of the image still translate to adding noise to the first set of tokens passed into the LLM
class NoiseRippleAdderPIL:
    """
    Add Gaussian noise to K patches starting at a given patch index (raster order) with a HARD cutoff.
    Optional 'ripple' = decay of noise strength across those K patches (no spill outside).
    """

    def __init__(
        self,
        mean=0.0,
        std=10.0,
        patch_size_px=14,
        start_patch=0,
        num_patches=1,
        clip=True,
        ripple=False,
        ripple_decay=0.8,  # strength multiplier per step (0<decay<=1)
        seed=None,
    ):
        self.mean = float(mean)
        self.std = float(std)
        self.patch_size_px = int(patch_size_px)
        self.start_patch = int(start_patch)
        self.num_patches = int(num_patches)
        self.clip = bool(clip)
        self.ripple = bool(ripple)
        self.ripple_decay = float(ripple_decay)
        self.rng = np.random.default_rng(seed)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        arr = np.array(image).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]  # (H,W,1)

        H, W, C = arr.shape
        ps = self.patch_size_px
        grid_h = H // ps
        grid_w = W // ps
        total = grid_h * grid_w

        if total <= 0 or self.num_patches <= 0:
            return image

        start = max(0, min(self.start_patch, total))
        end = max(start, min(start + self.num_patches, total))

        out = arr.copy()

        for idx in range(start, end):
            r = idx // grid_w
            c = idx % grid_w
            y0, y1 = r * ps, (r + 1) * ps
            x0, x1 = c * ps, (c + 1) * ps

            strength = 1.0
            if self.ripple:
                # decay relative to the start patch so ripple is local to the selected block
                strength = self.ripple_decay ** (idx - start)

            patch = out[y0:y1, x0:x1, :]
            noise = self.rng.normal(self.mean, self.std, size=patch.shape).astype(np.float32)
            patch = patch + strength * noise

            if self.clip:
                patch = np.clip(patch, 0.0, 255.0)

            out[y0:y1, x0:x1, :] = patch

        if out.shape[2] == 1:
            out = out[..., 0]

        return Image.fromarray(out.astype(np.uint8))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, "
            f"patch_size_px={self.patch_size_px}, start_patch={self.start_patch}, "
            f"num_patches={self.num_patches}, clip={self.clip}, ripple={self.ripple}, "
            f"ripple_decay={self.ripple_decay})"
        )