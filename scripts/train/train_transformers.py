from pathlib import Path
import argparse
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, DetrFeatureExtractor, DeformableDetrFeatureExtractor
import warnings

from apriorics.models import Detr, DeformableDetr, YOLOS 
from apriorics.data import CocoDetectionDetr, CocoDetectionDeformableDetr, CocoDetectionYOLOS

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help="Choose a model between detr, deformabledetr, yolos",
                    choices=['detr', 'deformabledetr', 'yolos'])
args = parser.parse_args()

data_path = Path('/data/elliot/coco_format_short_test/')
train_path = data_path / 'train'
val_path = data_path / 'val'

model_name = args.model 

def collate_fn_yolos(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['labels'] = labels
  return batch

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

if model_name == 'detr':
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    
    train_dataset = CocoDetectionDetr(img_folder=train_path / 'images', feature_extractor=feature_extractor)
    val_dataset = CocoDetectionDetr(img_folder=val_path / 'images', feature_extractor=feature_extractor, train=False)
        
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, num_workers = 8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, num_workers = 8, batch_size=1)
    
    model = Detr(lr=1e-5, lr_backbone=1e-4, weight_decay=1e-4, val_dataset=val_dataset)



elif model_name == 'deformabledetr':
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained("SenseTime/deformable-detr")
    
    train_dataset = CocoDetectionDeformableDetr(img_folder=train_path / 'images', feature_extractor=feature_extractor)
    val_dataset = CocoDetectionDeformableDetr(img_folder=val_path / 'images', feature_extractor=feature_extractor, train=False)
        
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, num_workers = 8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, num_workers = 8, batch_size=1)
    
    model = DeformableDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, val_dataset=val_dataset)



elif model_name == 'yolos':
    feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)
    
    train_dataset = CocoDetectionYOLOS(img_folder=train_path / 'images', feature_extractor=feature_extractor)
    val_dataset = CocoDetectionYOLOS(img_folder=val_path / 'images', feature_extractor=feature_extractor, train=False)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_yolos, batch_size=2, num_workers = 8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_yolos, num_workers = 8, batch_size=1)
    
    model = YOLOS(lr=1e-5, weight_decay=1e-4, val_dataset=val_dataset)



batch = next(iter(train_dataloader))

logger = CometLogger(
      api_key=os.environ["COMET_API_KEY"],
      workspace="apriorics",
      save_dir='/data/elliot/{}/'.format(model_name),
      project_name="apriorics",
      auto_metric_logging=False,
  )

if model_name == 'yolos':
    outputs = model(pixel_values=batch['pixel_values'])
else:
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

trainer = Trainer(gpus=1,
    gradient_clip_val=0.1,
    min_epochs=20,
    max_epochs=20,
    logger=logger,
    profiler='pytorch',
    default_root_dir="/data/elliot/{}/".format(model_name)
)

trainer.fit(
    model,
    train_dataloader,
    val_dataloader
)