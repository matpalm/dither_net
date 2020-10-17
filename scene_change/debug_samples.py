#!/usr/bin/env python3

# show samples of the scene boundary by showing a collage of the scene change
# frame as well as two frames before and after

import argparse
from PIL import Image
import util as u
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
opts = parser.parse_args()

manifest = u.read_manifest(opts.manifest_file, has_scene_change=True)

for idx in range(len(manifest)):
    if manifest[idx].scene_change and idx > 2:
        print("!", manifest[idx])
        images = []
        for window in range(idx-2, idx+3):
            print(" ", manifest[window].fname)
            images.append(Image.open(manifest[window].fname))
        collage_fname = "scene_change_samples/%s" % os.path.basename(
            manifest[idx].fname)
        u.collage(images).save(collage_fname)
