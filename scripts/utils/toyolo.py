import os
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Flip, RandomBrightnessContrast, RandomRotate90, Transpose
from pathaia.util.paths import get_files
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall, Specificity
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import maskrcnn_resnet50_fpn

from apriorics.data import BalancedRandomSampler, DetectionDataset, SparseDetectionDataset
from apriorics.metrics import DiceScore
from apriorics.plmodules import BasicDetectionModule, get_scheduler_func
from apriorics.transforms import (
    CorrectCompression,
    FixedCropAroundMaskIfExists,
    RandomCropAroundMaskIfExists,
    StainAugmentor,
    ToTensor,
)

from dataset_opening import dataset_opening
from PIL import Image as im

def output_yolo(path, dataset, split):
    for idx, patch in enumerate(dataset):
        patch_nb = idx
        slide_idx = dataset.slide_id[idx]
        image = im.fromarray(patch[0])
        print("{}-{}".format(patch_nb, slide_idx))
        with open(str(path) + "labels/{}/{}-{}.txt".format(split ,patch_nb, slide_idx), 'w') as f:
            for box in patch[1]["boxes"]:
                x0=box[0]
                y0=box[1]
                x1=box[2]
                y1=box[3]
                w= x1-x0
                h= y1-y0
                cx = (x0+x1)/2
                cy=(y0+y1)/2
                label = "{label} {cx} {cy} {w} {h}".format(label=0, cx=cx/256, cy=cy/256, w=w/256, h=h/256)
                f.write(label)
        image.save(str(path) + "images/{}/{}-{}.jpg".format(split, patch_nb, slide_idx))


path = Path("/data/elliot/yolo_dataset_emptyslides/")

print("train")
assert(os.path.exists(str(path) + "labels/train"))
assert(os.path.exists(str(path) + "images/train"))
train_ds = dataset_opening('train')
output_yolo(path, train_ds, 'train')


#print("valid")
assert(os.path.exists(str(path) + "labels/valid"))
assert(os.path.exists(str(path) + "images/valid"))
#val_ds = dataset_opening('val')
#output_yolo(path, val_ds, 'valid')
