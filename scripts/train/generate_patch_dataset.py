import csv
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pathaia.patches import filter_thumbnail
from pathaia.patches.functional_api import slide_rois_no_image
from pathaia.util.paths import get_files
from pathaia.util.types import Patch, Slide
from scipy.sparse import load_npz
from skimage.morphology import remove_small_objects

parser = ArgumentParser(prog="Generates the PathAIA patch CSVs for slides.")
parser.add_argument(
    "--slidefolder",
    type=Path,
    help="Input folder containing slide svs files.",
    required=True,
)
parser.add_argument(
    "--maskfolder",
    type=Path,
    help="Input folder containing mask tif files.",
    required=True,
)
parser.add_argument(
    "--outfolder",
    type=Path,
    help=(
        "Target output folder. Actual output folder will be "
        "outfolder/{patch_size}_{level}/patch_csvs."
    ),
    required=True,
)
parser.add_argument(
    "--recurse",
    "-r",
    action="store_true",
    help="Specify to recurse through slidefolder when looking for svs files. Optional.",
)
parser.add_argument(
    "--ihc-type",
    help="Name of the IHC.",
    required=True,
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
    "--patch-size",
    type=int,
    default=1024,
    help="Size of the (square) patches to extract. Default 1024.",
)
parser.add_argument(
    "--level",
    type=int,
    default=0,
    help="Pyramid level to extract patches on. Default 0.",
)
parser.add_argument(
    "--overlap",
    type=float,
    default=0,
    help="Part of the patches that should overlap. Default 0.",
)
parser.add_argument(
    "--filter-pos",
    type=int,
    default=0,
    help="Minimum number of positive pixels in mask to keep patch. Default 0.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Specify to overwrite existing csvs. Optional.",
)
parser.add_argument(
    "--num-workers",
    type=int,
    help="Number of workers to use for processing. Defaults to all available workers.",
)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    input_files = get_files(
        args.slidefolder / args.ihc_type / "HE",
        extensions=args.slide_extension,
        recurse=args.recurse,
    )
    input_files.sort(key=lambda x: x.stem)

    outfolder = (
        args.outfolder / args.ihc_type / f"{args.patch_size}_{args.level}/patch_csvs"
    )

    interval = -int(args.overlap * args.patch_size)
    if not outfolder.exists():
        outfolder.mkdir(parents=True)

    def write_patches(in_file_path):
        out_file_path = outfolder / in_file_path.with_suffix(".csv").name
        if not args.overwrite and out_file_path.exists():
            return

        mask_path = args.maskfolder / in_file_path.relative_to(
            args.slidefolder
        ).with_suffix(args.mask_extension)
        if not mask_path.exists():
            return

        slide = Slide(in_file_path, backend="cucim")
        if args.mask_extension == ".tif":
            mask = Slide(mask_path, backend="cucim")
        elif args.mask_extension == ".npz":
            mask = load_npz(mask_path)
        print(in_file_path.stem)

        patches = slide_rois_no_image(
            slide,
            args.level,
            psize=args.patch_size,
            interval=interval,
            slide_filters=[filter_thumbnail],
            thumb_size=2000,
        )

        with open(out_file_path, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=Patch.get_fields() + ["n_pos"])
            writer.writeheader()
            for patch in patches:
                if isinstance(mask, Slide):
                    patch_mask = np.asarray(
                        mask.read_region(
                            patch.position, patch.level, patch.size
                        ).convert("1")
                    )
                else:
                    w, h = patch.size
                    x, y = patch.position
                    patch_mask = mask[y : y + h, x : x + w].toarray()

                if args.filter_pos and patch_mask.sum():
                    patch_mask = remove_small_objects(
                        patch_mask, min_size=args.filter_pos
                    )
                row = patch.to_csv_row()
                n_pos = patch_mask.sum()
                row["n_pos"] = n_pos
                if n_pos >= args.filter_pos:
                    writer.writerow(row)

    with Pool(processes=args.num_workers) as pool:
        pool.map(write_patches, input_files)
        pool.close()
        pool.join()
