import json
from argparse import ArgumentParser
from pathlib import Path

import geopandas
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations import Crop
from metrics_config import METRICS
from pathaia.util.paths import get_files
from pytorch_lightning.utilities.seed import seed_everything
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from skimage.morphology import remove_small_holes
from timm import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from apriorics.data import get_dataset_cls
from apriorics.masks import flood_full_mask
from apriorics.metrics import MetricCollection
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import BasicClassificationModule, BasicSegmentationModule
from apriorics.polygons import mask_to_polygons_layer
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
    "--model",
    help=(
        "Model to use for training. If unet, can be formatted as unet/encoder to "
        "specify a specific encoder. Must be one of unet, med_t, logo, axalunet, gated."
    ),
    required=True,
)
parser.add_argument(
    "--ihc_type",
    choices=IHCS,
    help=f"Name of the IHC to train for. Must be one of {', '.join(IHCS)}.",
    required=True,
)
parser.add_argument(
    "--outfolder", type=Path, help="Output folder for geojsons.", required=True
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
    help="Input folder containing mask files.",
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
    "--batch_size",
    type=int,
    default=8,
    help=(
        "Batch size for training. effective batch size is multiplied by the number of"
        " gpus. Default 8."
    ),
)
parser.add_argument(
    "--patch_size",
    type=int,
    default=1024,
    help="Size of the patches used for training. Default 1024.",
)
parser.add_argument(
    "--base_size",
    type=int,
    default=1024,
    help=(
        "Size of the patches used before crop for training. Must be greater or equal "
        "to patch_size. Default 1024."
    ),
)
parser.add_argument(
    "--level", type=int, default=0, help="WSI level for patch extraction. Default 0."
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers to use for data loading. Default 0 (only main process).",
)
parser.add_argument(
    "--group_norm",
    action="store_true",
    help="Specify to use group norm instead of batch norm in model. Optional.",
)
parser.add_argument(
    "--hash_file",
    help="Yaml file containing the hash (=version) of the model to load weights from.",
    required=True,
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
    "--area_threshold",
    type=int,
    default=50,
    help="Minimum area of objects to keep. Default 50.",
)
parser.add_argument(
    "--slide_extension",
    default=".svs",
    help="File extension of slide files. Default .svs.",
)
parser.add_argument(
    "--mask_extension",
    default=".tif",
    help="File extension of mask files. Default .tif.",
)
parser.add_argument("--fold", default="0", help="Fold used for validation. Default 0.")
parser.add_argument(
    "--test_fold", default="test", help="Fold to use for test. Default test."
)
parser.add_argument(
    "--data_type",
    choices=["segmentation", "segmentation_sparse", "detection"],
    default="segmentation",
    help=(
        "Input data type. Must be one of segmentation, segmentation_sparse, "
        "detection. Default segmentation."
    ),
)
parser.add_argument("--classif-model")
parser.add_argument("--classif-version")
parser.add_argument("--flood-mask", action="store_true")


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    seed_everything(workers=True)

    trainfolder = args.trainfolder / args.ihc_type
    patch_csv_folder = trainfolder / f"{args.base_size}_{args.level}/patch_csvs"
    slidefolder = args.slidefolder / args.ihc_type / "HE"
    maskfolder = args.maskfolder / args.ihc_type / "HE"
    logfolder = args.trainfolder / "logs"

    with open(args.hash_file, "r") as f:
        version = yaml.safe_load(f)[args.fold]

    patches_paths = get_files(
        patch_csv_folder, extensions=".csv", recurse=False
    ).sorted(key=lambda x: x.stem)
    slide_paths = patches_paths.map(
        lambda x: slidefolder / x.with_suffix(args.slide_extension).name
    )
    mask_paths = patches_paths.map(
        lambda x: maskfolder / x.with_suffix(args.mask_extension).name
    )

    split_df = pd.read_csv(
        args.trainfolder / args.ihc_type / args.splitfile
    ).sort_values("slide")
    split_df = split_df.loc[split_df["slide"].isin(patches_paths.map(lambda x: x.stem))]
    val_idxs = (split_df["split"] == args.test_fold).values

    model = args.model.split("/")
    if model[0] == "unet":
        encoder_name = model[1]
    else:
        encoder_name = None

    device = torch.device(f"cuda:{args.gpu}")
    model = create_model(
        model[0],
        encoder_name=encoder_name,
        pretrained=True,
        img_size=args.patch_size,
        num_classes=1,
        norm_layer=group_norm if args.group_norm else torch.nn.BatchNorm2d,
    ).eval()
    model.requires_grad_(False)

    model = BasicSegmentationModule(
        model,
        loss=None,
        dl_lengths=(0, 0),
        lr=0,
        wd=0,
    ).to(device)

    ckpt_path = logfolder / f"apriorics/{version}/checkpoints/last.ckpt"
    checkpoint = torch.load(ckpt_path)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)

    if args.classif_model is not None and args.classif_version is not None:
        clf = create_model(
            args.classif_model, num_classes=1, norm_layer=torch.nn.BatchNorm2d
        ).eval()
        clf.requires_grad_(False)

        clf = BasicClassificationModule(
            clf, loss=None, dl_lengths=(0, 0), lr=0, wd=0
        ).to(device)

        ckpt_path = (
            logfolder / f"apriorics/{args.classif_version}/checkpoints/last.ckpt"
        )
        checkpoint = torch.load(ckpt_path)
        missing, unexpected = clf.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
    else:
        clf = None

    metrics = METRICS["all"]
    if args.ihc_type in METRICS:
        metrics.extend(METRICS[args.ihc_type])
    metrics = MetricCollection(metrics).to(device)

    dataset_cls = get_dataset_cls(args.data_type)

    if args.patch_size < args.base_size:
        interval = int(0.3 * args.patch_size)
        max_coord = args.base_size - args.patch_size
        crops = []
        for x in range(0, max_coord + 1, interval):
            for y in range(0, max_coord + 1, interval):
                crops.append((x, y, x + args.patch_size, y + args.patch_size))
            if max_coord % interval != 0:
                crops.append((x, max_coord, x + args.patch_size, args.base_size))
                crops.append((max_coord, x, args.base_size, x + args.patch_size))
        if max_coord % interval != 0:
            crops.append((max_coord, max_coord, args.base_size, args.base_size))
    else:
        crops = [(0, 0, args.base_size, args.base_size)]

    all_metrics = {}
    for slide_path, mask_path, patches_path in zip(
        slide_paths[val_idxs], mask_paths[val_idxs], patches_paths[val_idxs]
    ):
        print(slide_path.stem)
        polygons = []
        for crop in crops:
            print(crop)
            ds = dataset_cls(
                [slide_path],
                [mask_path],
                [patches_path],
                transforms=[Crop(*crop), ToTensor()],
            )
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            for batch_idx, (x, y) in tqdm(enumerate(dl), total=len(dl)):
                x = x.to(device)
                y_hat = torch.sigmoid(model(x))
                if clf is not None:
                    y_hat *= torch.sigmoid(clf(x))[:, None, None]
                metrics(y_hat, y.int().to(device), x=x)
                x = np.ascontiguousarray(
                    (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255),
                    dtype=np.uint8,
                )
                y_hat = y_hat.cpu().numpy() > 0.5
                for k, mask in enumerate(y_hat):
                    if args.flood_mask:
                        img = x[k]
                        mask = flood_full_mask(
                            img, mask, n=20, area_threshold=args.area_threshold
                        )
                    mask = remove_small_holes(mask, area_threshold=args.area_threshold)
                    if not mask.sum():
                        continue
                    idx = batch_idx * args.batch_size + k
                    patch = ds.patches[idx]
                    polygon = mask_to_polygons_layer(mask, angle_th=0, distance_th=0)
                    polygon = translate(
                        polygon,
                        xoff=patch.position.x + crop[0],
                        yoff=patch.position.y + crop[1],
                    )

                    if (
                        isinstance(polygon, Polygon)
                        and polygon.area > args.area_threshold
                    ):
                        polygons.append(polygon)
                    elif isinstance(polygon, MultiPolygon):
                        for pol in polygon.geoms:
                            if pol.area > args.area_threshold:
                                polygons.append(pol)

        polygons = unary_union(polygons)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])

        suffix = "_clf" if clf is not None else ""
        outfolder = args.outfolder / args.ihc_type / f"{version}{suffix}" / "geojsons"
        if not outfolder.exists():
            outfolder.mkdir(parents=True)

        with open(outfolder / f"{slide_path.stem}.geojson", "w") as f:
            json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)

        metrics_results = metrics.compute()
        metrics_results = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics_results.items()
        }

        outfolder = outfolder.parent / "metrics"
        if not outfolder.exists():
            outfolder.mkdir()

        with open(outfolder / f"{slide_path.stem}.json", "w") as f:
            json.dump(metrics_results, f)

        for k, v in metrics_results.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)
        metrics.reset()

    all_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    with open(outfolder.parent / "average_metrics.json", "w") as f:
        json.dump(all_metrics, f)
