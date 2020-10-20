#!/usr/bin/env python3

from joblib import Parallel, delayed
import argparse
import util as u
import os
from PIL import Image

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
opts = parser.parse_args()
print(opts)

u.ensure_dir_exists(opts.output_dir)


def recrop(fname):
    img = Image.open(os.path.join(opts.input_dir, fname))
    h, w = img.size
    img = img.crop((80, 80, h-80, w-80))
    img.save(os.path.join(opts.output_dir, fname))


Parallel(n_jobs=16)(delayed(recrop)(f) for f in os.listdir(opts.input_dir))
