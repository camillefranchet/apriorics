import datetime
import os
from math import ceil
from pathlib import Path

import pandas as pd
import torch
from albumentations import Flip, RandomBrightnessContrast, RandomRotate90, Transpose
from pathaia.util.paths import get_files
from pytorch_lightning.utilities.seed import seed_everything

from apriorics.data import BalancedRandomSampler, DetectionDataset, SparseDetectionDataset
from apriorics.plmodules import BasicDetectionModule, get_scheduler_func
from apriorics.transforms import (
    CorrectCompression,
    FixedCropAroundMaskIfExists,
    RandomCropAroundMaskIfExists,
    StainAugmentor,
    ToTensor,
)

from dataset_opening import dataset_opening

from PIL import Image

from pycococreator.pycococreatortools.pycococreatortools import create_image_info, create_annotation_info
import glob
import imageio
import numpy as np
from PIL import Image

import json 


# Adapted from : https://patrickwasp.com/create-your-own-coco-style-dataset/
def output_coco(path, dataset, s):

    print("currently treating :", s)

    coco_output = dict()


    coco_output['info'] = {
        "description": "Mitosis dataset",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "Elliot",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    coco_output['licenses'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]
    coco_output['categories'] = [
        {
            'id': 0,
            'name': 'Mitosis',
            'supercategory': 'N/A',
        }
    ]


    for idx, patch in enumerate(dataset):
        im = Image.fromarray(patch[0])
        slide_id = dataset.slide_id[idx]
        im.save(path / 'images' / "{}-{}.jpeg".format(slide_id, idx))
        for idx_mask, mask in enumerate(patch[1]['masks']):
            mk = Image.fromarray(mask)
            mk = mk.convert("L")
            mk.save(path / 'annotations' / "{}-{}_mitosis_{}.png".format(slide_id, idx, idx_mask))

     # filter for jpeg images
    IMAGE_DIR = path / 'images'
    ANNOTATION_DIR = path / 'annotations'

    coco_output['images'] = []
    coco_output['annotations'] = []
    segmentation_id = 0

    for image_id, image_filename in enumerate(glob.glob(str(IMAGE_DIR / '*.jpeg'))):
        image = Image.open(image_filename)
        slide_id = image_filename.split('.')[0].split('/')[-1]
        image_info = create_image_info(image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)

        for annotation_filename in (glob.glob(str(ANNOTATION_DIR / '{}_mitosis_*.png').format(slide_id))):
            if 'mitosis' in annotation_filename:
                class_id = 0

            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(imageio.imread(annotation_filename)).astype(np.uint8)
            
            annotation_info = create_annotation_info(segmentation_id, image_id, category_info, binary_mask,image.size, tolerance=2)
            segmentation_id += 1
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
        
    with open(IMAGE_DIR / "custom_{}.json".format(s), "w") as outfile:
        json.dump(coco_output, outfile)


train_ds = dataset_opening('train')

val_ds = dataset_opening('val')


sets = ['train', 'val']
data_path = Path('/data/elliot/coco_format_test_newscript/')
train_path = data_path / 'train'
val_path = data_path / 'val'

dic_path = dict()
dic_dataset = dict()

dic_path['train'] = train_path
dic_path['val'] = val_path

dic_dataset['train'] = train_ds
dic_dataset['val'] = val_ds
for s in sets:
    output_coco(dic_path[s], dic_dataset[s], s)