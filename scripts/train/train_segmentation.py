import os
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import horovod.torch as hvd
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from albumentations import (
    CenterCrop,
    Flip,
    RandomBrightnessContrast,
    RandomCrop,
    RandomRotate90,
    Transpose,
)
from metrics_config import METRICS
from pathaia.util.paths import get_files
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.seed import seed_everything
from timm import create_model
from torch.utils.data import DataLoader

from apriorics.data import (
    BalancedRandomSampler,
    ValidationPositiveSampler,
    get_dataset_cls,
)
from apriorics.losses import get_loss
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import BasicSegmentationModule, get_scheduler_func
from apriorics.stain_augment import StainAugmentor
from apriorics.transforms import ToTensor

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

parser = ArgumentParser(
    prog=(
        "Train a segmentation model for a specific IHC. To train on multiple gpus, "
        "should be called as `horovodrun -np n_gpus python train_segmentation.py "
        "--horovod`."
    )
)
parser.add_argument(
    "--hash-file",
    type=Path,
    help="File to store comet experiment version hash in. Optional",
)
parser.add_argument(
    "--model",
    help=(
        "Model to use for training. If unet, can be formatted as unet/encoder to "
        "specify a specific encoder. Must be one of unet, med_t, logo, axalunet, gated."
    ),
    required=True,
)
parser.add_argument(
    "--ihc-type",
    choices=IHCS,
    help=f"Name of the IHC to train for. Must be one of {', '.join(IHCS)}.",
    required=True,
)
parser.add_argument(
    "--trainfolder",
    type=Path,
    help="Folder containing all train files.",
    required=True,
)
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing svs slide files.",
    required=True,
)
parser.add_argument(
    "--maskfolder",
    type=Path,
    help="Input folder containing tif mask files.",
    required=True,
)
parser.add_argument(
    "--splitfile",
    type=str,
    default="splits.csv",
    help="Name of the csv file containing train/valid/test splits. Default splits.csv.",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU index to used when not using horovod. Default 0.",
)
parser.add_argument(
    "--horovod",
    action="store_true",
    help="Specify when using script with horovodrun. Optional.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help=(
        "Batch size for training. effective batch size is multiplied by the number of"
        " gpus. Default 8."
    ),
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for training. Default 1e-3."
)
parser.add_argument(
    "--wd",
    type=float,
    default=1e-2,
    help="Weight decay for AdamW optimizer. Default 1e-2.",
)
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train on. Default 10."
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=1024,
    help="Size of the input patches used during training. Default 1024.",
)
parser.add_argument(
    "--base-size",
    type=int,
    default=1024,
    help="Size of the original extracted patches. Default 1024.",
)
parser.add_argument(
    "--level", type=int, default=0, help="WSI level for patch extraction. Default 0."
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers to use for data loading. Default 0 (only main process).",
)
parser.add_argument(
    "--freeze-encoder",
    action="store_true",
    help="Specify to freeze encoder when using unet model. Optional.",
)
parser.add_argument(
    "--loss",
    default="bce",
    help=(
        "Loss function to use for training. Must be one of bce, focal, dice, "
        "sum_loss1_coef1_****. Default bce."
    ),
)
parser.add_argument(
    "--group-norm",
    action="store_true",
    help="Specify to use group norm instead of batch norm in model. Optional.",
)
parser.add_argument(
    "--scheduler",
    choices=["one-cycle", "cosine-anneal", "reduce-on-plateau"],
    help=(
        "Learning rate scheduler to use during training. Must be one of one-cycle, "
        "cosine-anneal, reduce-on-plateau. Optional."
    ),
)
parser.add_argument(
    "--grad-accumulation",
    type=int,
    default=1,
    help="Number of batches to accumulate gradients on. Default 1.",
)
parser.add_argument(
    "--resume-version", help="Version id of a model to load weights from. Optional."
)
parser.add_argument(
    "--seed",
    type=int,
    help=(
        "Specify seed for RNG. Can also be set using PL_GLOBAL_SEED environment "
        "variable. Optional."
    ),
)
parser.add_argument(
    "--augment-stain",
    action="store_true",
    help="Specify to use stain augmentation. Optional.",
)
parser.add_argument(
    "--slide-extension",
    default=".svs",
    help="File extension of slide files. Default .svs.",
)
parser.add_argument(
    "--mask-extension",
    default=".tif",
    help="File extension of slide files. Default .svs.",
)
parser.add_argument(
    "--fold", default="0", help="Fold to use for validation. Default 0."
)
parser.add_argument(
    "--p-pos",
    type=float,
    default=0.9,
    help=(
        "Percentage of masks containing positive pixels to use for training. Default "
        "0.9."
    ),
)
parser.add_argument(
    "--data-step",
    type=int,
    default=1,
    help="Give a step n > 1 to load one every n patches only. Default 1.",
)
parser.add_argument(
    "--data-type",
    choices=["segmentation", "segmentation_sparse", "detection"],
    default="segmentation",
    help=(
        "Input data type. Must be one of segmentation, segmentation_sparse, "
        "detection. Default segmentation."
    ),
)
parser.add_argument(
    "--grad-clip", type=float, help="Value to use for gradient clipping. Optional."
)

if __name__ == "__main__":
    __spec__ = None
    args = parser.parse_known_args()[0]
    if args.horovod:
        hvd.init()

    seed_everything(workers=True)

    trainfolder = args.trainfolder / args.ihc_type
    patch_csv_folder = trainfolder / f"{args.base_size}_{args.level}/patch_csvs"
    maskfolder = args.maskfolder / args.ihc_type / "HE"
    slidefolder = args.slidefolder / args.ihc_type / "HE"
    logfolder = args.trainfolder / "logs"

    patches_paths = get_files(
        patch_csv_folder, extensions=".csv", recurse=False
    ).sorted(key=lambda x: x.stem)
    mask_paths = patches_paths.map(
        lambda x: maskfolder / x.with_suffix(args.mask_extension).name
    )
    slide_paths = mask_paths.map(
        lambda x: slidefolder / x.with_suffix(args.slide_extension).name
    )

    split_df = pd.read_csv(
        args.trainfolder / args.ihc_type / args.splitfile
    ).sort_values("slide")
    split_df = split_df.loc[split_df["slide"].isin(patches_paths.map(lambda x: x.stem))]
    val_idxs = (split_df["split"] == args.fold).values
    train_idxs = ~val_idxs

    transforms = [
        RandomCrop(args.patch_size, args.patch_size),
        Flip(),
        Transpose(),
        RandomRotate90(),
        RandomBrightnessContrast(),
        ToTensor(),
    ]

    dataset_cls = get_dataset_cls(args.data_type)
    train_ds = dataset_cls(
        slide_paths[train_idxs],
        mask_paths[train_idxs],
        patches_paths[train_idxs],
        transforms=transforms,
        step=args.data_step,
    )
    val_ds = dataset_cls(
        slide_paths[val_idxs],
        mask_paths[val_idxs],
        patches_paths[val_idxs],
        transforms=[CenterCrop(args.patch_size, args.patch_size), ToTensor()],
        step=args.data_step,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
        sampler=BalancedRandomSampler(train_ds, p_pos=args.p_pos),
    )

    num_samples = int(1 / args.p_pos * (val_ds.n_pos > 0).sum())
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=ValidationPositiveSampler(val_ds, num_samples=num_samples),
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    scheduler_func = get_scheduler_func(
        args.scheduler,
        total_steps=ceil(len(train_dl) / (args.grad_accumulation)) * args.epochs,
        lr=args.lr,
    )

    model = args.model.split("/")
    if model[0] == "unet":
        encoder_name = model[1]
    else:
        encoder_name = None
    model = create_model(
        model[0],
        encoder_name=encoder_name,
        pretrained=True,
        img_size=args.patch_size,
        num_classes=1,
        norm_layer=group_norm if args.group_norm else torch.nn.BatchNorm2d,
    )

    metrics = METRICS["all"]
    if args.ihc_type in METRICS:
        metrics.extend(METRICS[args.ihc_type])
    plmodule = BasicSegmentationModule(
        model,
        loss=get_loss(args.loss),
        lr=args.lr,
        wd=args.wd,
        scheduler_func=scheduler_func,
        metrics=metrics,
        stain_augmentor=StainAugmentor() if args.augment_stain else None,
    )

    if args.freeze_encoder:
        plmodule.freeze_encoder()

    logger = CometLogger(
        api_key=os.environ["COMET_API_KEY"],
        workspace="apriorics",
        save_dir=logfolder,
        project_name="apriorics",
        auto_metric_logging=False,
    )

    if not args.horovod or hvd.rank() == 0:
        logger.experiment.add_tag(args.ihc_type)

    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        save_last=True,
        mode="min",
        filename="{epoch}-{val_loss:.3f}",
    )

    trainer = pl.Trainer(
        gpus=1 if args.horovod else [args.gpu],
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        logger=logger,
        precision=16,
        accumulate_grad_batches=args.grad_accumulation,
        callbacks=[ckpt_callback],
        strategy="horovod" if args.horovod else None,
        num_sanity_val_steps=0,
        gradient_clip_val=args.grad_clip,
    )

    if args.resume_version is not None:
        ckpt_path = logfolder / f"apriorics/{args.resume_version}/checkpoints/last.ckpt"
        checkpoint = torch.load(ckpt_path)
        missing, unexpected = plmodule.load_state_dict(
            checkpoint["state_dict"], strict=False
        )

    try:
        trainer.fit(
            plmodule,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            # ckpt_path=ckpt_path,
        )
    finally:
        if args.hash_file is not None and trainer.current_epoch:
            with open(args.hash_file, "w") as f:
                yaml.dump({args.fold: logger.version}, f)
