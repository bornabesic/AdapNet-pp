import argparse
from pathlib import Path
import random

import h5py
from tqdm import tqdm
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, nargs="+")
parser.add_argument("n_images", type=int)
parser.add_argument("output_directory", type=str)
args = parser.parse_args()

assert all(map(lambda p: Path(p).exists(), args.file))
output = Path(args.output_directory)
output_rgb = output / "rgb"
output_semseg = output / "semseg"
output_rgb.mkdir(exist_ok=True, parents=True)
output_semseg.mkdir(exist_ok=True, parents=True)

output_paths = output / "paths.txt"

handles = {file_path: h5py.File(file_path, "r", libver="latest") for file_path in args.file}
n_images_total = sum(map(lambda hdf5: hdf5["frames"].shape[0], handles.values()))
print(n_images_total)
with output_paths.open("wt") as paths:
    for hdf5 in handles.values():
        frames = hdf5["frames"]
        idxs = list(range(frames.shape[0]))
        random.shuffle(idxs)
        n = int(round(frames.shape[0] / n_images_total * args.n_images))
        for idx in tqdm(idxs[:n]):
            rgb = Image.fromarray(frames[idx]["rgb_dynamic"])
            semseg = Image.fromarray(frames[idx]["semseg_dynamic"])
            rgb.save(output_rgb / f"{idx:05}.png")
            semseg.save(output_semseg / f"{idx:05}.png")
            print(str((output_rgb / f"{idx:05}.png").resolve()), str((output_semseg / f"{idx:05}.png").resolve()), file=paths)
        hdf5.close()
