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
import re
import tqdm
import data as d

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=str)
parser.add_argument('--ckpt-idx', type=int)
parser.add_argument('--manifest', type=str, help='list of files to process')
parser.add_argument('--output-dir', type=str)
parser.add_argument('--batch-size', type=int, default=8)
opts = parser.parse_args()
print(opts)

u.ensure_dir_exists(opts.output_dir)

generator = g.Generator()
generator = objax.Jit(generator)
ckpt = objax.io.Checkpoint(logdir=f"ckpts/{opts.run}/generator", keep_ckpts=10)
ckpt.restore(generator.vars(), idx=opts.ckpt_idx)


def load_rgb(fname):
    rgb = Image.open(fname)
    rgb = np.array(rgb, dtype=np.float32)
    rgb /= 255.
    return rgb


def load_dither(fname):
    rgb = Image.open(fname)
    dither = rgb.convert(mode='1', dither=Image.FLOYDSTEINBERG)
    dither = np.array(dither, dtype=np.float32)
    dither = np.expand_dims(dither, -1)
    return dither


def fname_rgb_t1_dither_t0():
    fnames = list(map(str.strip, open(opts.manifest).readlines()))
    t0_idxs = list(range(len(fnames)-1))
    for t0_idx in t0_idxs:
        t1_idx = t0_idx + 1
        dither_t0 = load_dither(fnames[t0_idx])
        rgb_t1 = load_rgb(fnames[t1_idx])
        yield fnames[t1_idx], rgb_t1, dither_t0


def just_fname(full_name):
    m = re.match(".*/(f_.*jpg)", full_name)
    if not m:
        raise Exception("unexpected filename [%s]" % full_name)
    return m.group(1)


dataset = (tf.data.Dataset.from_generator(fname_rgb_t1_dither_t0,
                                          output_types=(tf.string, tf.float32,
                                                        tf.float32))
           .batch(opts.batch_size)
           .prefetch(1))

for fnames, rgb_imgs_t1, dither_t0 in dataset:
    fnames = fnames.numpy()
    rgb_imgs_t1 = rgb_imgs_t1.numpy()
    dither_t0 = dither_t0.numpy()

    pred_dithers = generator(rgb_imgs_t1, dither_t0)
    for dither, full_fname in zip(pred_dithers, fnames):
        dest_fname = just_fname(full_fname.decode()).replace("jpg", "png")
        dither_pil = u.dither_to_pil(dither)
        dither_pil = u.center_crop(dither_pil, 1448, 1072)
        dither_pil.save("%s/%s" % (opts.output_dir, dest_fname))
