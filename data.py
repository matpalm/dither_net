
import glob
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import util as u
from functools import lru_cache

# TODO: pack to TFRECORD eventually...


@lru_cache(None)
def parse(fname):
    rgb_img = Image.open(fname)
    true_dither = rgb_img.convert(mode='1', dither=Image.FLOYDSTEINBERG)
    rgb_img = np.array(rgb_img, dtype=np.float32)
    rgb_img /= 255.
    true_dither = np.array(true_dither, dtype=np.float32)
    true_dither = np.expand_dims(true_dither, -1)  # single channel
    return rgb_img, true_dither


def dataset(fname_glob, batch_size):
    def fnames():
        fnames = glob.glob(fname_glob)
        random.shuffle(fnames)
        for fname in fnames:
            yield parse(fname)

    return (tf.data.Dataset.from_generator(fnames,
                                           output_types=(tf.float32, tf.float32))
            .batch(batch_size)
            .prefetch(AUTOTUNE))


# for rgb, td in dataset(batch_size=4):
#     print(rgb.shape, td.shape)
#     u.dither_to_pil_image(td[0]).show()
#     break
