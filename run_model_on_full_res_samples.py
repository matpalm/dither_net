#!/usr/bin/env python3

import objax
import models
import data
import numpy as np
from PIL import Image
import argparse
import util as u

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


predict = objax.Jit(predict, unet.vars())


def parse(fname):


def fnames():
    frame = 3000
    while frame <= 162999:
        yield "frames/full_res/f_%08d.jpg" % frame
        frame += 1000


dataset = (tf.data.Dataset.from_generator(fnames, output_types=(tf.string))
           .map(parse, AUTOTUNE)
           .batch(16))
for fname in fnames():
    print(fname)

    # parse RGB and true dither
    rgb_img, true_dither = data.parse_full_size(fname)

    # run rgb through network to get predicted dither
    pred_dither = predict(np.expand_dims(rgb_img, 0))[0]

    # make a collage of the three images
    w, h = rgb_img.shape[1], rgb_img.shape[0]
    collage = Image.new('RGB', (w*3, h))
    collage.paste(u.rgb_img_to_pil(rgb_img), (0, 0))
    collage.paste(u.dither_to_pil(true_dither), (w, 0))
    collage.paste(u.dither_to_pil(pred_dither), (w*2, 0))
    collage.save("collages/%08d.png" % frame)
