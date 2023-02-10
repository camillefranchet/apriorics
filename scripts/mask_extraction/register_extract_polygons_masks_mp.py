from argparse import ArgumentParser
from ctypes import c_bool
from functools import partial
from multiprocessing import Array, Lock, Pool, current_process
from pathlib import Path
from subprocess import run

import docker
import numpy as np
from ordered_set import OrderedSet
from pathaia.patches import slide_rois_no_image
from pathaia.util.paths import get_files
from pathaia.util.types import Coord, Patch, Slide
from PIL import Image
from shapely.affinity import translate
from shapely.ops import unary_union

from apriorics.masks import get_mask_function, get_tissue_mask, update_full_mask_mp
from apriorics.polygons import mask_to_polygons_layer
from apriorics.registration import full_registration, get_coord_transform

parser = ArgumentParser(
    prog=(
        "Registers IHC slides to corresponding H&E slides using HistoReg and extracts"
        " masks by thresholding DAB on IHC."
    )
)
parser.add_argument(
    "--dab_thr",
    type=float,
    default=0.03,
    help=(
        "Threshold to use for basic DAB thresholding. Only purpose is to skip "
        "registration for some patches when not enough DAB is present. Default 0.03."
    ),
)
parser.add_argument(
    "--object_min-size",
    type=int,
    default=1000,
    help=(
        "Minimum object size (in pixels) to keep on mask. Only purpose is to skip "
        "registration for some patches when not enough DAB is present. Default 1000."
    ),
)
parser.add_argument(
    "--binary_op",
    default="closing",
    choices=["none", "closing", "dilation"],
    help=(
        "Scikit-image binary operation to use on mask. Only purpose is to skip "
        "registration for some patches when not enough DAB is present. Must be one of "
        "closing, dilation, none. Default closing."
    ),
)
parser.add_argument(
    "--radius",
    default=10,
    type=int,
    help="Radius of the disk to use as footprint for binary operation. Default 10.",
)
parser.add_argument(
    "--psize",
    type=int,
    default=5000,
    help=(
        "Size of the patches that are used for registration and mask extraction. "
        "Default 5000."
    ),
)
parser.add_argument(
    "--overlap",
    type=float,
    default=0.3,
    help="Part of the patches that should overlap. Default 0.3.",
)
parser.add_argument(
    "--crop",
    type=float,
    default=0.1,
    help=(
        "Part of the patches to crop for mask extraction (to avoid registration "
        "artifacts). Default 0.1."
    ),
)
parser.add_argument(
    "--data_path",
    type=Path,
    help="Main data folder containing all input and output subfolders.",
    required=True,
)
parser.add_argument(
    "--slide_path",
    type=Path,
    help="Input folder that contains input svs slide files.",
    required=True,
)
parser.add_argument(
    "--ihc_type",
    help=("Name of the IHC to extract masks from."),
    required=True,
)
parser.add_argument(
    "--tmp_path",
    type=Path,
    default="tmp",
    help=(
        "Path to the temporary folder that will be used for computation. Default tmp."
    ),
)
parser.add_argument("--mask_path", type=Path, help="Output mask folder.", required=True)
parser.add_argument("--wkt_path", type=Path, help="Output wkt folder.", required=True)
parser.add_argument(
    "--novips",
    action="store_true",
    help=(
        "Specify to avoid converting masks from png to pyramidal tiled tif. Useful "
        "when vips is not installed. Optional."
    ),
)
parser.add_argument(
    "--num_workers",
    type=int,
    help="Number of workers to use for processing. Defaults to all available workers.",
)
parser.add_argument(
    "--slide_extension", default=".svs", help="Extension of slide files. Default .svs."
)


def get_filefilter(slidefile):
    if slidefile is not None:
        with open(slidefile, "r") as f:
            files = set(f.read().rstrip().split("\n"))
    else:
        files = None

    def _filter(items):
        names = OrderedSet(items.map(lambda x: x.stem))
        if files is not None:
            return names.index(names & files)
        else:
            return list(range(len(names)))

    return _filter


def pool_init_func(arg_slide_he, arg_slide_ihc, arg_full_mask):
    global slide_he
    global slide_ihc
    global full_mask
    slide_he = arg_slide_he
    slide_ihc = arg_slide_ihc
    full_mask = arg_full_mask


def register_extract_mask(args, patch_he):
    crop = int(args.crop * args.psize)
    box = (crop, crop, args.psize - crop, args.psize - crop)

    tmpfolder = args.data_path / args.tmp_path

    try:
        coord_tfm = get_coord_transform(slide_he, slide_ihc)
    except IndexError:

        def coord_tfm(x, y):
            return Coord(x, y)

    patch_ihc = Patch(
        id=patch_he.id,
        slidename="",
        position=coord_tfm(*patch_he.position),
        size=patch_he.size,
        level=patch_he.level,
        size_0=patch_he.size_0,
    )

    pid = str(current_process().pid)
    base_path = tmpfolder / pid
    if not base_path.exists():
        base_path.mkdir()

    restart = True
    iterations = 5000
    count = 0
    maxiter = 4

    while restart and count < maxiter:
        container = full_registration(
            slide_he,
            slide_ihc,
            patch_he,
            patch_ihc,
            base_path,
            dab_thr=args.dab_thr,
            object_min_size=args.object_min_size,
            iterations=iterations,
            threads=1,
        )
        restart = container is False
        if restart:
            break
        else:
            try:
                ihc = Image.open(base_path / "ihc_warped.png").convert("RGB").crop(box)
            except FileNotFoundError:
                print(f"[{pid}] ERROR: HE={patch_he}/IHC={patch_ihc}. Restarting...")
                container.stop()
                container.remove()
                return
            tissue_mask = get_tissue_mask(np.asarray(ihc.convert("L")), whitetol=256)
            if tissue_mask.sum() < 0.999 * tissue_mask.size:
                container.stop()
                container.remove()
                restart = True
                iterations *= 2
                count += 1

    if restart:
        return

    print(f"[{pid}] Computing mask...")

    he = Image.open(base_path / "he.png")
    he = np.asarray(he.convert("RGB").crop(box))
    mask = get_mask_function(args.ihc_type)(he, np.asarray(ihc))
    update_full_mask_mp(
        full_mask, mask, *(patch_he.position + crop), *slide_he.dimensions
    )
    polygons = mask_to_polygons_layer(mask, 0, 0)
    x, y = patch_he.position
    moved_polygons = translate(polygons, x + crop, y + crop)

    res = container.exec_run("rm -rf /data/*", stream=True)

    with open(base_path / "log", "ab") as f:
        for chunk in res.output:
            f.write(chunk)

    container.stop()
    container.remove()

    print(f"[{pid}] Mask done.")
    return moved_polygons


def main(args):
    slidefolder = args.data_path / args.slide_path
    maskfolder = args.data_path / args.mask_path
    wktfolder = args.data_path / args.wkt_path
    tmpfolder = args.data_path / args.tmp_path

    if not maskfolder.exists():
        maskfolder.mkdir()

    if not wktfolder.exists():
        wktfolder.mkdir()

    if not tmpfolder.exists():
        tmpfolder.mkdir()

    interval = -int(args.overlap * args.psize)

    hefiles = get_files(
        slidefolder / args.ihc_type / "HE",
        extensions=args.slide_extension,
        recurse=False,
    )
    hefiles.sort(key=lambda x: x.stem.split("-")[0])
    ihcfiles = get_files(
        slidefolder / args.ihc_type / "IHC",
        extensions=args.slide_extension,
        recurse=False,
    )
    ihcfiles.sort(key=lambda x: x.stem.split("-")[0])

    henames = OrderedSet(hefiles.map(lambda x: x.stem.split("-")[0]))
    ihcnames = OrderedSet(ihcfiles.map(lambda x: x.stem.split("-")[0]))
    inter = henames & ihcnames
    heidxs = henames.index(inter)
    ihcidxs = ihcnames.index(inter)
    hefiles = hefiles[heidxs]
    ihcfiles = ihcfiles[ihcidxs]

    for hefile, ihcfile in zip(hefiles, ihcfiles):
        hefile = Path(hefile)
        ihcfile = Path(ihcfile)

        maskpath = maskfolder / hefile.relative_to(slidefolder).with_suffix(".png")
        if not maskpath.parent.exists():
            maskpath.parent.mkdir(parents=True)

        if maskpath.with_suffix(".tif").exists() or (maskpath).exists():
            continue

        print(hefile, ihcfile)

        slide_he = Slide(slidefolder / hefile, backend="cucim")
        slide_ihc = Slide(slidefolder / ihcfile, backend="cucim")
        w, h = slide_he.dimensions

        lock = Lock()
        full_mask = Array(c_bool, h * w, lock=lock)
        _register_extract_mask = partial(register_extract_mask, args)

        with Pool(
            processes=args.num_workers,
            initializer=pool_init_func,
            initargs=(slide_he, slide_ihc, full_mask),
        ) as pool:
            patch_iter = slide_rois_no_image(
                slide_he,
                0,
                (args.psize, args.psize),
                (interval, interval),
                thumb_size=5000,
                slide_filters=["full"],
            )
            all_polygons = pool.map(_register_extract_mask, patch_iter)
            pool.close()
            pool.join()

        all_polygons = [x for x in all_polygons if x is not None]
        all_polygons = unary_union(all_polygons)

        print("Saving full polygons...")

        wktfile = wktfolder / hefile.relative_to(slidefolder).with_suffix(".wkt")
        if not wktfile.parent.exists():
            wktfile.parent.mkdir(parents=True)
        with wktfile.open("w") as f:
            f.write(all_polygons.wkt)

        print("Polygons saved.")

        print("Saving mask...")

        full_mask_np = np.frombuffer(full_mask.get_obj(), dtype=bool).reshape(h, w)
        Image.fromarray(full_mask_np).convert("RGB").save(maskpath)
        if not args.novips:
            vips_cmd = (
                f"vips tiffsave {maskpath} {maskpath.with_suffix('.tif')} "
                "--compression jpeg --Q 100 --tile-width 256 --tile-height 256 --tile "
                "--pyramid"
            )

            run(vips_cmd.split())

            maskpath.unlink()

        print("Mask saved.")

    client = docker.from_env()
    client.containers.run(
        "historeg",
        f"rm -rf /data/{tmpfolder.name}",
        volumes=[f"{tmpfolder.parent}:/data"],
    )


if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    main(args)
