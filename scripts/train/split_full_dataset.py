import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from apriorics.dataset_preparation import split_data_k_fold

parser = ArgumentParser(prog="Splits a dataset between train and validation slides. ")
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
    "--test_ratio",
    type=float,
    default=0.1,
    help="Part of the dataset to use for test. Default 0.1.",
)
parser.add_argument(
    "--nfolds", type=int, default=5, help="Number of folds to create. Default 5."
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
    "--locked_test_file",
    type=Path,
    help="Path to a text file containing all slide names reserved for test.",
)
parser.add_argument("--min_slide", type=int, default=4)
parser.add_argument("--max_slide", type=int, default=1288)

if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    previous_splits = {}

    if args.locked_test_file is not None:
        with open(args.locked_test_file, "r") as f:
            test_slides = f.read().rstrip().split("\n")
        previous_splits["test"] = np.array(test_slides)

    if args.update and args.out_csv.exists():
        ex_df = pd.read_csv(args.out_csv)

        for i in ex_df["split"].unique():
            split_slides = ex_df.loc[ex_df["split"] == i, "slide"].values
            if i == "test":
                split_slides = set(split_slides) | set(previous_splits["test"])
                split_slides = np.array(list(split_slides))
            previous_splits[i] = split_slides

    if args.seed is None:
        args.seed = os.environ.get("PL_GLOBAL_SEED")
        if args.seed is not None:
            args.seed = int(args.seed)

    slidenames = [f"21I{n:06d}" for n in range(args.min_slide, args.max_slide + 1)]
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
