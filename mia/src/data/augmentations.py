import torch
import torchvision.transforms as transforms
from functools import partial
from PIL import Image

def get_augmentations(cfg):
    aug_dict = cfg.data.augmentations
    aug_func_dict = dict()
    for key, values in aug_dict.items():
        if key == "RandomResize" and values.use == True:
            _aug_f_list = list()
            for _size, _scale, _ratio in zip(values.size, values.scale, values.ratio):
                aug_f = transforms.RandomResizedCrop(
                    size=_size,
                    scale=_scale,
                    ratio=_ratio
                )
                _aug_f_list.append(aug_f)
            aug_func_dict[key] = _aug_f_list
        elif key == "RandomRotation" and values.use == True:
            _aug_f_list = list()
            for _degrees in values.degrees:
                aug_f = transforms.RandomRotation(
                    degrees=_degrees
                )
                _aug_f_list.append(aug_f)
            aug_func_dict[key] = _aug_f_list
        elif key == "RandomAffine" and values.use == True:
            _aug_f_list = list()
            for _degree, _translate, _scale in zip(values.degrees, values.translate, values.scale):
                aug_f = transforms.RandomAffine(
                    degrees=_degree,
                    translate=_translate,
                    scale=_scale
                )
                _aug_f_list.append(aug_f)
            aug_func_dict[key] = _aug_f_list
        elif key == "ColorJitter" and values.use == True:
            _aug_f_list = list()
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
            aug_func_dict[key] = _aug_f_list
        elif key == "GaussianNoise" and values.use == True:
            _aug_f_list = list()
            for _mean, _std in zip(values.mean, values.std):
                aug_f = AddGaussianNoisePIL(
                    mean=_mean,
                    std=_std,
                    clip=True
                )
                _aug_f_list.append(aug_f)
            aug_func_dict[key] = _aug_f_list
            
    return aug_func_dict

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