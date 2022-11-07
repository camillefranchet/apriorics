import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from geopandas import GeoSeries
from pathaia.util.paths import get_files
from pathaia.util.types import Slide
from scipy.sparse import csr_array, save_npz
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from skimage.morphology import label, remove_small_holes, remove_small_objects
from tqdm import tqdm

from apriorics.masks import flood_mask, get_mask_ink, remove_large_objects
from apriorics.polygons import mask_to_polygons_layer

parser = ArgumentParser()
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--maskfolder", type=Path)
parser.add_argument("--geojsonfolder", type=Path)
parser.add_argument("--outmaskfolder", type=Path)
parser.add_argument("--outgeojsonfolder", type=Path)
parser.add_argument("--recurse", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    maskpaths = get_files(args.maskfolder, extensions=".tif", recurse=args.recurse)
    for maskpath in maskpaths:
        outmaskpath = args.outmaskfolder / maskpath.relative_to(
            args.maskfolder
        ).with_suffix(".npz")
        if not outmaskpath.parent.exists():
            outmaskpath.parent.mkdir(parents=True)
        outgeojsonpath = args.outgeojsonfolder / maskpath.relative_to(
            args.maskfolder
        ).with_suffix(".geojson")
        if not outgeojsonpath.parent.exists():
            outgeojsonpath.parent.mkdir(parents=True)

        if outmaskpath.exists() and outgeojsonpath.exists():
            continue

        slidename = maskpath.stem
        print(slidename)
        slidepath = args.slidefolder / maskpath.relative_to(
            args.maskfolder
        ).with_suffix(".svs")
        geojsonpath = args.geojsonfolder / maskpath.relative_to(
            args.maskfolder
        ).with_suffix(".geojson")

        with open(geojsonpath, "r") as f:
            geojson = json.load(f)
            target_pols = []
            for feat in geojson["features"]:
                geom = feat["geometry"]
                if geom["type"] == "Polygon":
                    target_pols.append(shape(geom))
        target_pols = MultiPolygon(target_pols)

        slide = Slide(slidepath, backend="cucim")
        mask = Slide(maskpath, backend="cucim")
        w, h = mask.dimensions

        new_pols = []
        n = 20
        d = 100
        row_inds = []
        col_inds = []
        for pol in tqdm(target_pols.geoms):
            x0, y0, x1, y1 = map(int, pol.bounds)
            c = pol.representative_point()
            slide_reg = np.asarray(
                slide.read_region(
                    (x0 - d, y0 - d), 0, (2 * d + x1 - x0, 2 * d + y1 - y0)
                ).convert("RGB")
            )
            mask_reg = np.asarray(
                mask.read_region(
                    (x0 - d, y0 - d), 0, (2 * d + x1 - x0, 2 * d + y1 - y0)
                ).convert("1")
            )
            labels, n_obj = label(mask_reg, return_num=True)
            for i in range(n_obj):
                sub_mask = labels == i + 1
                if sub_mask[int(c.y) - y0 + d, int(c.x) - x0 + d]:
                    mask_reg = sub_mask
                    break
            mask_ink = get_mask_ink(slide_reg)
            if mask_ink.sum() > 5e-3 * mask_ink.size:
                continue

            out = flood_mask(slide_reg, mask_reg, n)  # & ~mask_ink
            out = remove_large_objects(
                remove_small_objects(
                    remove_small_holes(out, area_threshold=100), min_size=50
                ),
                max_size=2000,
            )
            ii, jj = out.nonzero()
            row_inds.extend((ii + y0 - d).tolist())
            col_inds.extend((jj + x0 - d).tolist())
            new_pol = mask_to_polygons_layer(out, 0, 0)
            new_pol = translate(new_pol, x0 - d, y0 - d)
            new_pols.append(new_pol)

        new_pols = unary_union(new_pols)
        if isinstance(new_pols, Polygon):
            new_pols = MultiPolygon([new_pols])

        sparse_mask = csr_array(
            ([1] * len(row_inds), (row_inds, col_inds)), shape=(h, w), dtype=bool
        )
        save_npz(outmaskpath, sparse_mask)

        with open(outgeojsonpath, "w") as f:
            json.dump(GeoSeries(new_pols.geoms).__geo_interface__, f)
