from argparse import ArgumentParser
from pathlib import Path

from openslide import OpenSlide
from pathaia.util.paths import get_files

from apriorics.polygons import hovernet_to_geojson

parser = ArgumentParser(prog="Converts json files outputted by hovernet to wkt.")
parser.add_argument(
    "--jsonfolder", type=Path, help="Input hovernet json folder.", required=True
)
parser.add_argument(
    "--geojsonfolder", type=Path, help="Output geojson folder.", required=True
)
parser.add_argument("--slidefolder", type=Path)
parser.add_argument("--crop-size", type=int)


if __name__ == "__main__":
    args = parser.parse_args()

    jsonfiles = get_files(args.jsonfolder, extensions=[".json"], recurse=False)

    if not args.geojsonfolder.exists():
        args.geojsonfolder.mkdir()

    for jsonfile in jsonfiles:
        slidename = ".".join(jsonfile.name.split(".")[:-1])
        geojsonfile = args.geojsonfolder / f"{slidename}.geojson"
        if geojsonfile.exists():
            continue
        print(slidename)
        if args.slidefolder is not None:
            slidefile = args.slidefolder / f"{slidename}.mrxs"
            slide = OpenSlide(str(slidefile))
            try:
                x = int(slide.properties["openslide.bounds-x"])
                y = int(slide.properties["openslide.bounds-y"])
            except KeyError:
                x = 0
                y = 0
        else:
            x = 0
            y = 0
        hovernet_to_geojson(
            jsonfile, geojsonfile, crop_size=args.crop_size, xoff=-x, yoff=-y
        )
