import json
import os
import shutil
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from pathaia.util.paths import get_files

from apriorics.data_utils import get_info_from_filename

parser = ArgumentParser()
parser.add_argument("--remote-path", type=Path, required=True)
parser.add_argument("--remote-rel-path", type=Path, required=True)
parser.add_argument("--data-path", type=Path, required=True)
parser.add_argument("--rel-path", type=Path, required=True)
parser.add_argument("--extension", default=".svs")
parser.add_argument("--import-ihc", action="store_true")
parser.add_argument("--clean-previous", action="store_true")
parser.add_argument("--ihc-type", required=True)
parser.add_argument("--mapping-file", type=Path, required=True)
parser.add_argument("--recurse", action="store_true")
parser.add_argument("--out-csv", type=Path)
parser.add_argument("--num-workers", type=int, default=os.cpu_count())
parser.add_argument("--add-tree", action="store_true")


if __name__ == "__main__":
    args = parser.parse_known_args()[0]

    remote_path = args.remote_path / args.remote_rel_path
    local_path = args.data_path / args.rel_path
    if not local_path.exists():
        local_path.mkdir(parents=True)

    with open(args.mapping_file, "r") as f:
        ihc_mapping = json.load(f)

    assert (
        args.ihc_type in ihc_mapping["HE"].keys()
    ), f"ihc-type must be one of {', '.join(ihc_mapping['HE'].keys())}"

    if args.clean_previous:
        for ihc in ihc_mapping["HE"].keys():
            if ihc != args.ihc_type and (local_path / ihc).exists():
                shutil.rmtree(local_path / ihc)

    files = get_files(remote_path, extensions=args.extension, recurse=args.recurse)

    def transfer_file(file):
        try:
            info = get_info_from_filename(file.stem, ihc_mapping)
        except ValueError:
            ihc_type = None
            slide_type = "HE"
            for part in file.parts:
                if part in ihc_mapping["HE"].keys():
                    ihc_type = part
                elif part == "IHC":
                    slide_type = "IHC"
            if ihc_type is None:
                raise ValueError("IHC type info not found in file path.")
            info = {
                "ihc_type": ihc_type,
                "slide_type": slide_type,
                "block": file.stem,
            }
        if not (
            info["ihc_type"] == args.ihc_type
            and (info["slide_type"] == "HE" or args.import_ihc)
        ):
            return

        if args.add_tree:
            outfolder = local_path / info["ihc_type"] / info["slide_type"]
        else:
            outfolder = local_path
        if not outfolder.exists():
            outfolder.mkdir(parents=True)

        outfile = outfolder / f"{info['block']}{file.suffix}"
        if not (outfile.exists() and outfile.stat().st_size == file.stat().st_size):
            print(outfile)
            shutil.copyfile(file, outfile)
        return file.stem, str(outfile)

    data = {"id": [], "path": []}
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for res in pool.map(transfer_file, files):
            if res is not None:
                data["id"].append(res[0])
                data["path"].append(res[1])

    if args.out_csv is not None:
        df = pd.DataFrame(data)
        df.to_csv(args.out_csv, index=False)
