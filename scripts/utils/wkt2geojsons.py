import json
from argparse import ArgumentParser
from pathlib import Path

import geopandas
from pathaia.util.paths import get_files
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import clip_by_rect

parser = ArgumentParser("Transforms wkt files into geojson format.")
parser.add_argument(
    "--data_path",
    type=Path,
    help="Main data folder containing all input and output subfolders.",
    required=True,
)
parser.add_argument(
    "--wkt_path", type=Path, help="Input folder containing wkt files.", required=True
)
parser.add_argument(
    "--geojson_path", type=Path, help="Output folder for geojson files.", required=True
)
parser.add_argument("--ihc_type", required=True)
parser.add_argument("--crop", nargs=4)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    wktfiles = get_files(
        args.data_path / args.wkt_path / args.ihc_type / "HE",
        extensions=".wkt",
        recurse=False,
    )

    for wktfile in wktfiles:
        outfile = (
            args.data_path
            / args.geojson_path
            / wktfile.relative_to(args.data_path / args.wkt_path).with_suffix(
                ".geojson"
            )
        )

        if outfile.exists():
            continue

        print(wktfile)

        with open(wktfile, "r") as f:
            polygons = wkt.load(f)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])

        if args.crop is not None:
            x0, y0, x1, y1 = args.crop
            polygons = clip_by_rect(polygons, x0, y0, x1, y1)

        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True)

        with open(outfile, "w") as f:
            json.dump(geopandas.GeoSeries(polygons.geoms).__geo_interface__, f)
