import cv2
from albumentations import (
    Affine,
    CenterCrop,
    Flip,
    GaussianBlur,
    GaussNoise,
    HueSaturationValue,
    MedianBlur,
    OneOf,
    RandomBrightnessContrast,
    RandomCrop,
    RandomRotate90,
    Transpose,
)

from apriorics.transforms import ToTensor


def get_transforms(name, crop_size):
    transforms = {
        "base": [
            RandomCrop(crop_size, crop_size),
            Flip(),
            Transpose(),
            RandomRotate90(),
            RandomBrightnessContrast(),
            ToTensor(),
        ],
        "hovernet": [
            Affine(
                scale=(0.8, 1.2),
                translate_percent=[-0.01, 0.01],
                shear=(-5, 5),
                rotate=(-179, 179),
                interpolation=cv2.INTER_NEAREST,
                p=1,
            ),
            CenterCrop(crop_size, crop_size),
            Flip(p=0.75),
            OneOf(
                [
                    GaussianBlur(blur_limit=(1, 7), p=1),
                    MedianBlur(blur_limit=7, p=1),
                    OneOf(
                        [
                            GaussNoise(var_limit=(0.05 * 255) ** 2, p=1),
                            GaussNoise(
                                var_limit=(0.05 * 255) ** 2, per_channel=False, p=1
                            ),
                        ],
                        p=1,
                    ),
                ],
                p=0.75,
            ),
            HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, p=1),
            RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=1),
            ToTensor(),
        ],
    }
    return transforms[name]
