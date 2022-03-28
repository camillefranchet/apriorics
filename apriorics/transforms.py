import random
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from albumentations import CropNonEmptyMaskIfExists
import numpy as np
from pathaia.util.basic import ifnone
from typing import Callable, Optional, Any, List, Sequence, Tuple, Dict
from nptyping import NDArray
from numbers import Number
from staintools.miscellaneous.optical_density_conversion import convert_RGB_to_OD
import spams
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from torchvision.transforms.functional import to_tensor
import torch
from pathaia.util.types import NDImage, NDGrayImage, NDByteImage
from skimage.morphology import label, remove_small_holes, remove_small_objects


class ToSingleChannelMask(DualTransform):
    """
    Transforms that takes a grayscale masks with rgb or rgba channels and transform them
    into a single channel image

    Target : mask, masks
    Type : any
    """

    def __init__(self, trailing_channels: bool = True):
        super().__init__(True, 1)
        self.trailing_channels = trailing_channels

    def apply(self, img: NDImage, **params) -> NDImage:
        return img

    def apply_to_mask(self, img: NDImage, **params) -> NDGrayImage:
        if self.trailing_channels:
            return img[:, :, 0]
        else:
            return img[0]


class DropAlphaChannel(DualTransform):
    """
    Transform that takes rgba images and mask and that removes the alpha channel

    Target : image, mask, masks
    Type : any
    """

    def __init__(self, trailing_channels: bool = True):
        super().__init__(True, 1)
        self.trailing_channels = trailing_channels

    def apply(self, img: NDImage, **params) -> NDImage:
        if self.trailing_channels:
            assert img.shape[2] == 4
            return img[:, :, :-1]
        else:
            assert img.shape[0] == 4
            return img[:-1]


class ToTensor(DualTransform):
    def __init__(
        self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> Dict[str, Callable[[NDImage], torch.Tensor]]:
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img: NDImage, **params) -> torch.Tensor:
        return to_tensor(img)

    def apply_to_mask(self, mask: NDImage, **params) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("transpose_mask",)


def get_concentrations(
    img: NDByteImage, stain_matrix: NDArray[(2, 3), float], regularizer: float = 0.01
) -> NDArray[(Any, Any, 2), float]:
    OD = convert_RGB_to_OD(img).reshape((-1, 3))
    HE = spams.lasso(
        X=OD.T,
        D=stain_matrix.T,
        mode=2,
        lambda1=regularizer,
        pos=True,
        numThreads=1,
    )
    return HE.toarray().T


class StainAugmentor(ImageOnlyTransform):
    def __init__(
        self,
        alpha_range: float = 0.3,
        beta_range: float = 0.2,
        alpha_stain_range: float = 0.3,
        beta_stain_range: float = 0.2,
        he_ratio: float = 0.3,
        always_apply: bool = True,
        p: float = 1,
    ):
        super(StainAugmentor, self).__init__(always_apply, p)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.alpha_stain_range = alpha_stain_range
        self.beta_stain_range = beta_stain_range
        self.he_ratio = he_ratio

    def get_params(self):
        return {
            "alpha": np.random.uniform(
                1 - self.alpha_range, 1 + self.alpha_range, size=2
            ),
            "beta": np.random.uniform(-self.beta_range, self.beta_range, size=2),
            "alpha_stain": np.stack(
                (
                    np.random.uniform(
                        1 - self.alpha_stain_range * self.he_ratio,
                        1 + self.alpha_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        1 - self.alpha_stain_range,
                        1 + self.alpha_stain_range,
                        size=3,
                    ),
                ),
            ),
            "beta_stain": np.stack(
                (
                    np.random.uniform(
                        -self.beta_stain_range * self.he_ratio,
                        self.beta_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        -self.beta_stain_range, self.beta_stain_range, size=3
                    ),
                ),
            ),
        }

    def initialize(
        self,
        alpha: Optional[NDArray[(Any, ...), float]],
        beta: Optional[NDArray[(Any, ...), float]],
        shape: Tuple[int, ...] = 2,
    ) -> Tuple[NDArray[(Any, ...)], NDArray[(Any, ...)]]:
        alpha = ifnone(alpha, np.ones(shape))
        beta = ifnone(beta, np.zeros(shape))
        return alpha, beta

    def apply(
        self,
        image_and_stain: Tuple[
            NDArray[(Any, Any, 3), Number], Optional[NDArray[(2, 3), float]]
        ],
        alpha: Optional[NDArray[(2,), float]] = None,
        beta: Optional[NDArray[(2,), float]] = None,
        alpha_stain: Optional[NDArray[(2, 3), float]] = None,
        beta_stain: Optional[NDArray[(2, 3), float]] = None,
        **params
    ) -> NDArray[(Any, Any, 3), Number]:
        image, stain_matrix = image_and_stain
        alpha, beta = self.initialize(alpha, beta, shape=2)
        alpha_stain, beta_stain = self.initialize(alpha_stain, beta_stain, shape=(2, 3))
        if stain_matrix is None:
            stain_matrix = VahadaneStainExtractor.get_stain_matrix(image)
        if not image.dtype == np.uint8:
            image = (image * 255).astype(np.uint8)

        HE = get_concentrations(image, stain_matrix)
        stain_matrix = stain_matrix * alpha_stain + beta_stain
        stain_matrix = np.clip(stain_matrix, 0, 1)
        HE = np.where(HE > 0.2, HE * alpha[None] + beta[None], HE)
        out = np.exp(-np.dot(HE, stain_matrix)).reshape(image.shape)
        out = np.clip(out, 0, 1)
        return out.astype(np.float32)

    def get_transform_init_args_names(self) -> List[str]:
        return (
            "alpha_range",
            "beta_range",
            "alpha_stain_range",
            "beta_stain_range",
            "he_ratio",
        )

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return params


class RandomCropAroundMaskIfExists(CropNonEmptyMaskIfExists):
    """Crop area with mask if mask is non-empty, else make random crop. Cropped area
    will always be centered around a non empty area with a random offset.
    Args:
        height: vertical size of crop in pixels
        width: horizontal size of crop in pixels
        ignore_values: values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        ignore_channels: channels to ignore in mask
            (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
        p: probability of applying the transform. Default: 1.0.
    """

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: Optional[Sequence[int]] = None,
        ignore_channels: Optional[Sequence[int]] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(RandomCropAroundMaskIfExists, self).__init__(
            height,
            width,
            ignore_values=ignore_values,
            ignore_channels=ignore_channels,
            always_apply=always_apply,
            p=p,
        )
        self.height = height
        self.width = width

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(masks[0])
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            labels, n = label(mask, return_num=True)
            idx = random.randint(1, n)
            mask = labels == idx
            non_zero_yx = np.argwhere(mask)
            ymin, xmin = non_zero_yx.min(0)
            ymax, xmax = non_zero_yx.max(0)
            x_min = random.randint(
                max(0, xmax - self.width), min(xmin, mask_width - self.width)
            )
            y_min = random.randint(
                max(0, ymax - self.width), min(ymin, mask_width - self.width)
            )
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params


class FixedCropAroundMaskIfExists(CropNonEmptyMaskIfExists):
    """Crop area with mask if mask is non-empty, else make center crop. Cropped area
    will always be centered around a non empty area in a fully deterministic way.
    Args:
        height: vertical size of crop in pixels
        width: horizontal size of crop in pixels
        ignore_values: values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        ignore_channels: channels to ignore in mask
            (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
        p: probability of applying the transform. Default: 1.0.
    """

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: Optional[Sequence[int]] = None,
        ignore_channels: Optional[Sequence[int]] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(FixedCropAroundMaskIfExists, self).__init__(
            height,
            width,
            ignore_values=ignore_values,
            ignore_channels=ignore_channels,
            always_apply=always_apply,
            p=p,
        )
        self.height = height
        self.width = width

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(masks[0])
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for FixedCropAroundMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            shape = np.array([self.height, self.width], dtype=np.int64)
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            labels, n = label(mask, return_num=True)
            mask = np.zeros_like(mask)
            for i in range(1, n + 1):
                if (labels == i).sum() > mask.sum():
                    mask = labels == i
            non_zero_yx = np.argwhere(mask)
            center = non_zero_yx.mean(axis=0, dtype=np.int64)
            y_min, x_min = np.maximum(center - shape, 0)
            y_min = min(y_min, mask_height - self.height)
            x_min = min(x_min, mask_width - self.width)
        else:
            y_min = (mask_height - self.height) // 2
            x_min = (mask_width - self.width) // 2

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params


class CorrectCompression(DualTransform):
    def __init__(
        self,
        min_size: int = 10,
        area_threshold: int = 10,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.min_size = min_size
        self.area_threshold = area_threshold

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        if mask.ndim == 3:
            new_mask = mask[:, :, 0] > 0
        else:
            new_mask = mask > 0
        new_mask = remove_small_objects(
            remove_small_holes(new_mask, area_threshold=self.area_threshold),
            min_size=self.min_size,
        )
        new_mask = new_mask.astype(mask.dtype)
        if mask.dtype == np.uint8:
            new_mask *= 255
        if mask.ndim == 3:
            new_mask = np.repeat(new_mask[:, :, None], mask.shape[-1], axis=-1)
        return new_mask

    def get_transform_init_args_names(self):
        return ("min_size", "area_threshold")
