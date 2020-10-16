

from PIL import Image
import numpy as np
import random
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import util as u
from functools import lru_cache


def parse_full_size(fname):
    rgb_img = Image.open(fname)
    true_dither = rgb_img.convert(mode='1', dither=Image.FLOYDSTEINBERG)
    rgb_img = np.array(rgb_img, dtype=np.float32)
    rgb_img /= 255.
    true_dither = np.array(true_dither, dtype=np.float32)
    true_dither = np.expand_dims(true_dither, -1)  # single channel
    return rgb_img, true_dither


def parse(fname, patch_size, crops_per_img=64):
    rgb_img, true_dither = parse_full_size(fname)
    w, h = rgb_img.shape[1], rgb_img.shape[0]
    for _ in range(crops_per_img):
        left = random.randint(0, h-patch_size)
        top = random.randint(0, w-patch_size)
        rgb_crop = rgb_img[left:left+patch_size, top:top+patch_size, :]
        dither_crop = true_dither[left:left +
                                  patch_size, top: top+patch_size, :]
        yield rgb_crop, dither_crop


def dataset(manifest_file, batch_size, patch_size, shuffle_buffer_size=4096):
    def crops():
        fnames = list(map(str.strip, open(manifest_file).readlines()))
        while True:
            random.shuffle(fnames)
            for fname in fnames:
                for crops in parse(fname, patch_size):
                    yield crops

    return (tf.data.Dataset.from_generator(crops,
                                           output_types=(tf.float32, tf.float32))
            .shuffle(shuffle_buffer_size)
            .batch(batch_size)
            .prefetch(AUTOTUNE))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fname', type=str)
    opts = parser.parse_args()
    rgb, dither = parse(opts.fname)
    rgb = u.rgb_img_to_pil(rgb)
    dither = u.dither_to_pil(dither)
    assert rgb.size == dither.size
    w, h = rgb.size
    collage = Image.new('RGB', (w*2, h))
    collage.paste(rgb, (0, 0))
    collage.paste(dither, (w, 0))
    collage.show()
