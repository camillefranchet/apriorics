import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pathaia.util.paths import get_files

from apriorics.dataset_preparation import split_data_k_fold

parser = ArgumentParser(prog="Splits a dataset between train and validation slides. ")
parser.add_argument(
    "--maskfolder",
    type=Path,
    help="Input folder containing mask files.",
    required=True,
)
parser.add_argument(
    "--out_csv",
    type=Path,
    help="Output csv containing 2 columns: slide and split.",
    required=True,
)
parser.add_argument(
    "--update",
    action="store_true",
    help=("Specify to update existing csv instead of completely recreating it."),
)
parser.add_argument(
    "--recurse",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)
parser.add_argument(
    "--test_ratio",
    type=float,
    default=0.1,
    help="Part of the dataset to use for test. Default 0.1.",
)
parser.add_argument(
    "--nfolds", type=int, default=5, help="Number of folds to create. Default 5."
)
parser.add_argument(
    "--ihc_type",
    help="Name of the IHC.",
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
parser.add_argument("--mask_extension", default=".tif")

if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    slidenames = get_files(
        args.maskfolder / args.ihc_type / "HE",
        extensions=args.mask_extension,
        recurse=args.recurse,
    ).map(lambda x: x.stem)

    if args.update and args.out_csv.exists():
        ex_df = pd.read_csv(args.out_csv)
        previous_splits = {}
        for i in ex_df["split"].unique():
            previous_splits[i] = ex_df.loc[ex_df["split"] == i, "slide"].values
    else:
        previous_splits = None

    if args.seed is None:
        args.seed = os.environ.get("PL_GLOBAL_SEED")
        if args.seed is not None:
            args.seed = int(args.seed)

    splits_dict = split_data_k_fold(
        slidenames,
        k=args.nfolds,
        test_size=args.test_ratio,
        seed=args.seed,
        previous_splits=previous_splits,
    )

    slides = []
    splits = []

    for k, v in splits_dict.items():
        splits.extend([k] * len(v))
        slides.append(v)
    slides = np.concatenate(slides)

    df = pd.DataFrame({"slide": slides, "split": splits}).sort_values(by="slide")

    df.to_csv(args.out_csv, index=False)
