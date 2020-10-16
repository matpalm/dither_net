#!/usr/bin/env python3

# show samples of the scene boundary by showing a collage
# of 2 frames before and 2 after

import argparse
from PIL import Image, ImageDraw
import util as u

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
opts = parser.parse_args()

manifest = u.read_manifest(opts.manifest_file, has_scene_change=True)

for fname, scene_change in manifest:
    print("f", fname, "sc", scene_change)
