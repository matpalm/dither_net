#!/usr/bin/env python3

import objax
import generator as g
import data
import numpy as np
from PIL import Image
import argparse
import util as u
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import sys
import re

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=str)
parser.add_argument('--ckpt-idx', type=int)
parser.add_argument('--manifest', type=str, help='list of files to process')
parser.add_argument('--output-dir', type=str)
parser.add_argument('--batch-size', type=int, default=8)
opts = parser.parse_args()
print(opts)

generator = g.Generator()
generator = objax.Jit(generator)
ckpt = objax.io.Checkpoint(logdir=f"ckpts/{opts.run}/generator", keep_ckpts=10)
ckpt.restore(generator.vars(), idx=opts.ckpt_idx)


def fnames():
    for fname in open(opts.manifest, 'r').readlines():
        yield fname.strip()


def parse(fname):
    img = tf.io.read_file(fname)
    img = tf.image.decode_png(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img, fname


def just_fname(full_name):
    m = re.match(".*/(f_.*jpg)", full_name)
    if not m:
        raise Exception("unexpected filename [%s]" % full_name)
    return m.group(1)


dataset = (tf.data.Dataset.from_generator(fnames, output_types=(tf.string))
           .map(parse, AUTOTUNE)
           .batch(opts.batch_size)
           .prefetch(1))

for rgb_imgs, fnames in dataset:
    rgb_imgs = rgb_imgs.numpy()
    fnames = fnames.numpy()
    pred_dithers = generator(rgb_imgs)
    for dither, full_fname in zip(pred_dithers, fnames):
        dest_fname = just_fname(full_fname.decode()).replace("jpg", "png")
        dither_pil = u.dither_to_pil(dither)
        dither_pil = u.center_crop(dither_pil, 1448, 1072)
        dither_pil.save("%s/%s" % (opts.output_dir, dest_fname))
    sys.stdout.write("%s                       \r" % dest_fname)
    sys.stdout.flush()
print()
