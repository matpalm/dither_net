

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


def parse(fname_t0, fname_t1, patch_size, crops_per_img=64):
    rgb_img_t0, true_dither_t0 = parse_full_size(fname_t0)
    rgb_img_t1, true_dither_t1 = parse_full_size(fname_t1)

    w, h = rgb_img_t0.shape[1], rgb_img_t0.shape[0]
    for _ in range(crops_per_img):
        left = random.randint(0, h-patch_size)
        top = random.randint(0, w-patch_size)

        def crop(img_array):
            return img_array[left:left+patch_size, top:top+patch_size, :]

        yield (crop(rgb_img_t0), crop(true_dither_t0),
               crop(rgb_img_t1), crop(true_dither_t1))


def dataset(manifest_file, batch_size, patch_size, shuffle_buffer_size=4096):
    def crops():
        fnames = list(map(str.strip, open(manifest_file).readlines()))
        t0_idxs = list(range(len(fnames)-1))
        while True:
            random.shuffle(t0_idxs)
            for t0_idx in t0_idxs:
                for crops in parse(fnames[t0_idx], fnames[t0_idx+1],
                                   patch_size):
                    yield crops

    return (tf.data.Dataset.from_generator(crops, output_types=(tf.float32,
                                                                tf.float32))
            .shuffle(shuffle_buffer_size)
            .batch(batch_size)
            .prefetch(AUTOTUNE))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fname-t0', type=str, required=True)
    parser.add_argument('--fname-t1', type=str, required=True)
    opts = parser.parse_args()
    crops = parse(opts.fname_t0, opts.fname_t1,
                  patch_size=128, crops_per_img=4)
    rgb_t0, dither_t0, rgb_t1, dither_t1 = next(crops)
    rgb_t0 = u.rgb_img_to_pil(rgb_t0)
    rgb_t1 = u.rgb_img_to_pil(rgb_t1)
    dither_t0 = u.dither_to_pil(dither_t0)
    dither_t1 = u.dither_to_pil(dither_t1)
    w, h = rgb_t0.size
    collage = Image.new('RGB', (w*2, h*2))
    collage.paste(rgb_t0, (0, 0))
    collage.paste(dither_t0, (w, 0))
    collage.paste(rgb_t1, (0, h))
    collage.paste(dither_t1, (w, h))
    collage.show()
