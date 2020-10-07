#!/usr/bin/env python3

import objax
import models
import data
import numpy as np
from PIL import Image
import argparse
import util as u
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import sys

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run', type=str)
parser.add_argument('--ckpt-idx', type=int)
opts = parser.parse_args()
print(opts)

unet = models.Unet()
ckpt = objax.io.Checkpoint(logdir=f"ckpts/{opts.run}/", keep_ckpts=10)
ckpt.restore(unet.vars(), idx=opts.ckpt_idx)


def predict(rgb_imgs):
    return unet(rgb_imgs, training=False)


def parse(fname):
    img = tf.io.read_file(fname)
    img = tf.image.decode_png(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img, fname


def fnames():
    frame = 9230
    while frame <= 162999:
        yield "frames/full_res/f_%08d.jpg" % frame
#        frame += 1000
        frame += 1


predict = objax.Jit(predict, unet.vars())

dataset = (tf.data.Dataset.from_generator(fnames, output_types=(tf.string))
           .map(parse, AUTOTUNE)
           .batch(16))

for rgb_imgs, fnames in dataset:
    rgb_imgs = rgb_imgs.numpy()
    fnames = fnames.numpy()
    pred_dithers = predict(rgb_imgs)
    for dither, fname in zip(pred_dithers, fnames):
        fname = fname.decode().replace("full_res", "full_res_dithers")
        fname = fname.decode().replace(".jpg", ".png")
        u.dither_to_pil(dither).save(fname)
    sys.stdout.write("%s                       \r" % fname)
    sys.stdout.flush()
