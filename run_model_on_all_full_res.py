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
parser.add_argument('--input-dir', type=str, default='frames/full_res/')
parser.add_argument('--output-dir', type=str)
opts = parser.parse_args()
print(opts)

generator = g.Generator()
ckpt = objax.io.Checkpoint(logdir=f"ckpts/{opts.run}/", keep_ckpts=10)
ckpt.restore(generator.vars(), idx=opts.ckpt_idx)

generator = objax.Jit(generator)


def fnames():
    frame = 9230
    while frame <= 162999:
        yield "%s/f_%08d.jpg" % (opts.input_dir, frame)
#        frame += 1000
        frame += 1


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
           .batch(16))

for rgb_imgs, fnames in dataset:
    rgb_imgs = rgb_imgs.numpy()
    fnames = fnames.numpy()
    pred_dithers = generator(rgb_imgs)
    for dither, full_fname in zip(pred_dithers, fnames):
        just_fname = just_fname(full_fname).replace("jpg", "png")
        u.dither_to_pil(dither).save("%s/%s" % (opts.output_dir, just_fname))
    sys.stdout.write("%s                       \r" % just_fname)
    sys.stdout.flush()
