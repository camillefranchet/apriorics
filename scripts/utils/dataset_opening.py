from math import ceil
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Flip, RandomBrightnessContrast, RandomRotate90, Transpose
from pathaia.util.paths import get_files
from pytorch_lightning.utilities.seed import seed_everything
from torchvision.models.detection import maskrcnn_resnet50_fpn
from apriorics.data import BalancedRandomSampler, DetectionDataset, SparseDetectionDataset
from apriorics.transforms import (
    CorrectCompression,
    FixedCropAroundMaskIfExists,
    RandomCropAroundMaskIfExists,
    StainAugmentor,
    ToTensor,
)

import time



IHCS = [
    "AE1AE3",
    "CD163",
    "CD3CD20",
    "EMD",
    "ERGCaldes",
    "ERGPodo",
    "INI1",
    "P40ColIV",
    "PHH3",
]

from xmlrpc.client import Boolean


class arguments():

    ihc_type = "PPH3"
    patch_csv_folder=Path("/data/anapath/AprioricsSlides/train/PHH3/256_0/patch_csvs/")
    slidefolder=Path("/data/anapath/AprioricsSlides/slides/PHH3/HE/")
    maskfolder=Path("/data/anapath/AprioricsSlides/masks/PHH3/HE/" )
    split_csv=Path("/data/anapath/AprioricsSlides/train/splits.csv")
    logfolder=Path("/data/anapath/AprioricsSlides/train/logs")
    gpu=0
    batch_size=32
    lr=2e-4
    wd=0.01
    epochs=10
    patch_size=224
    num_workers=32
    scheduler="one-cycle"
    horovod=False
    stain_matrices_folder=Path("")
    augment_stain=False

args = arguments()

if args.horovod:
    hvd.init()

seed_everything(workers=True)

print("data loading")

patches_paths = get_files(
    args.patch_csv_folder, extensions=".csv", recurse=False
).sorted(key=lambda x: x.stem)
mask_paths = patches_paths.map(
    lambda x: args.maskfolder / x.with_suffix(".npz").name
)
slide_paths = mask_paths.map(
    lambda x: args.slidefolder / x.with_suffix(".svs").name
)
print("data splitting")

split_df = pd.read_csv(args.split_csv).sort_values("slide")
train_idxs = (split_df["split"] != "test").values
val_idxs = ~train_idxs


train_idxs = train_idxs[0:3]
val_idxs = val_idxs[0:3]

if args.stain_matrices_folder is not None:
    stain_matrices_paths = mask_paths.map(
        lambda x: args.stain_matrices_folder / x.with_suffix(".npy").name
    )
    stain_matrices_paths = stain_matrices_paths[train_idxs]
else:
    stain_matrices_paths = None

print("transformers")

transforms = [
    #CorrectCompression(),
    RandomCropAroundMaskIfExists(args.patch_size, args.patch_size),
    Flip(),
    Transpose(),
    RandomRotate90(),
    RandomBrightnessContrast(),
    ToTensor(),
]

if torch.cuda.is_available():
    num_workers = 8
import time
start_time = time.time()

def dataset_opening(split):
    start_time = time.time()

    if split == 'train':
        print("train")
        ds = SparseDetectionDataset(
                slide_paths[train_idxs],
                mask_paths[train_idxs],
                patches_paths[train_idxs],
                #stain_matrices_paths,
                #stain_augmentor=StainAugmentor() if args.augment_stain else None,
                #transforms=transforms,
            )
    if split == 'val':
        ds = SparseDetectionDataset(
                slide_paths[val_idxs],
                mask_paths[val_idxs],
                patches_paths[val_idxs],
                #stain_matrices_paths,
                #stain_augmentor=StainAugmentor() if args.augment_stain else None,
                #transforms=transforms,
            )
    print("--- %s seconds ---" % (time.time() - start_time))
    return ds
