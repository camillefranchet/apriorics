from apriorics.polygons import hovernet_to_wkt
from pathaia.util.paths import get_files
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(prog="Converts json files outputted by hovernet to wkt.")
parser.add_argument("--jsonfolder", type=Path, help="Input hovernet json folder.")
parser.add_argument("--wktfolder", type=Path, help="Output wkt folder.")


if __name__ == "__main__":
    args = parser.parse_args()

    jsonfiles = get_files(args.jsonfolder, extensions=[".json"], recurse=False)

    if not args.wktfolder.exists():
        args.wktfolder.mkdir()

    for jsonfile in jsonfiles:
        wktfile = args.wktfolder / f"{jsonfile.stem}.wkt"
        hovernet_to_wkt(jsonfile, wktfile)
