import json
from argparse import ArgumentParser
from pathlib import Path

import geopandas
import pandas as pd
import torch
from albumentations import Crop
from metrics_config import METRICS
from pathaia.util.paths import get_files
from pytorch_lightning.utilities.seed import seed_everything
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from timm import create_model
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from apriorics.data import SegmentationDataset
from apriorics.model_components.normalization import group_norm
from apriorics.plmodules import BasicSegmentationModule
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
    "--ihc-type",
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
    "--batch-size",
    type=int,
    default=8,
    help=(
        "Batch size for training. effective batch size is multiplied by the number of"
        " gpus. Default 8."
    ),
)
parser.add_argument(
    "--patch-size",
    type=int,
    default=1024,
    help="Size of the patches used for training. Default 1024.",
)
parser.add_argument(
    "--base-size",
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
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers to use for data loading. Default 0 (only main process).",
)
parser.add_argument(
    "--group-norm",
    action="store_true",
    help="Specify to use group norm instead of batch norm in model. Optional.",
)
parser.add_argument(
    "--version", help="Version id of a model to load weights from.", required=True
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
    "--area-threshold",
    type=int,
    default=50,
    help="Minimum area of objects to keep. Default 50.",
)
parser.add_argument(
    "--slide-extension",
    default=".svs",
    help="File extension of slide files. Default .svs.",
)
parser.add_argument(
    "--mask-extension",
    default=".tif",
    help="File extension of mask files. Default .tif.",
)
parser.add_argument(
    "--fold", default="test", help="Fold to use for test. Default test."
)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    seed_everything(workers=True)

    trainfolder = args.trainfolder / args.ihc_type
    patch_csv_folder = trainfolder / f"{args.base_size}_{args.level}/patch_csvs"
    slidefolder = args.slidefolder / args.ihc_type / "HE"
    maskfolder = args.maskfolder / args.ihc_type / "HE"
    logfolder = args.trainfolder / "logs"

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
    val_idxs = (split_df["split"] == args.fold).values

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

    metrics = METRICS["all"]
    if args.ihc_type in METRICS:
        metrics.append(METRICS[args.ihc_type])
    metrics = MetricCollection(metrics)

    model = BasicSegmentationModule(
        model,
        loss=None,
        lr=0,
        wd=0,
    ).to(device)

    ckpt_path = logfolder / f"apriorics/{args.version}/checkpoints/last.ckpt"
    checkpoint = torch.load(ckpt_path)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)

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

    for slide_path, patches_path in zip(slide_paths[val_idxs], patches_paths[val_idxs]):
        print(slide_path.stem)
        polygons = []
        for crop in crops:
            print(crop)
            ds = SegmentationDataset(
                slide_path,
                mask_paths,
                patches_path,
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
                y_hat = torch.sigmoid(model(x.to(device)))
                metrics(y_hat, y.int())
                y_hat = y_hat.cpu().numpy() > 0.5
                for k, mask in enumerate(y_hat):
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
        metrics_results = metrics.compute()
        metrics_results = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics_results.items()
        }

        polygons = unary_union(polygons)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])

        outfolder = args.outfolder / args.ihc_type / args.version / "geojsons"
        if not outfolder.exists():
            outfolder.mkdir(parents=True)

        with open(outfolder.parent / "metrics.json", "w") as f:
            json.dump(metrics_results, f)

        with open(outfolder / f"{slide_path.stem}.geojson", "w") as f:
            json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)
