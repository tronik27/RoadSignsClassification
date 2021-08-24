import albumentations as aug
from typing import Tuple


def image_augmentation(config: list, target_shape: Tuple[int, int, int]):
    """
    Adds gaussian blur to images.
    :param config: list with a set of parameters for augmentation.
    :param target_shape: tuple with input shape of image.
    """
    augmentation_list = []
    augmentations = {'vertical_flip': aug.VerticalFlip(),
                     'horizontal_flip': aug.HorizontalFlip(),
                     'sharpen': aug.Sharpen(alpha=(0.7, 1.0), lightness=(0.5, 0.7)),
                     'rgb_shift': aug.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                     'brightness_contrast': aug.RandomBrightnessContrast(brightness_limit=0.05,
                                                                         contrast_limit=(0, 0.25)),
                     'hue_saturation': aug.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20),
                     'distortion': aug.OneOf([aug.OpticalDistortion(), aug.GridDistortion(distort_limit=0.07)]),
                     'noise': aug.OneOf([aug.ISONoise(intensity=(0.1, 0.25)), aug.GaussNoise(var_limit=(10.0, 25.0)),
                                         aug.MultiplicativeNoise()]),
                     'blur': aug.OneOf([aug.GaussianBlur(blur_limit=(1, 3)), aug.GlassBlur(max_delta=2),
                                        aug.MedianBlur(blur_limit=3)]),
                     'crop': aug.RandomResizedCrop(height=target_shape[0], width=target_shape[1], scale=(0.75, 1.0)),
                     'rotate': aug.geometric.rotate.Rotate(limit=10)}
    for augmentation in config:
        augmentation_list.append(augmentations[augmentation])
    transform = aug.Compose(augmentation_list)
    return transform
