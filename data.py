

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


def parse_t0_t1(fname_t0, fname_t1, patch_size, crops_per_img=64):
    # parse RGB and dither for two frames; used to train generator
    # returns rgb_t1, dither_t0, dither_t1
    # note * we don't use rgb_t0,
    #      * and D doesn't need dither_t0

    _rgb_img_t0, true_dither_t0 = parse_full_size(fname_t0)
    rgb_img_t1, true_dither_t1 = parse_full_size(fname_t1)

    w, h = rgb_img_t1.shape[1], rgb_img_t1.shape[0]
    for _ in range(crops_per_img):
        left = random.randint(0, h-patch_size)
        top = random.randint(0, w-patch_size)

        def crop(img_array):
            return img_array[left:left+patch_size, top:top+patch_size, :]

        yield crop(rgb_img_t1), crop(true_dither_t0), crop(true_dither_t1)


def dataset(manifest_file, batch_size, patch_size, shuffle_buffer_size=4096):
    def crops():
        fnames = list(map(str.strip, open(manifest_file).readlines()))
        t0_idxs = list(range(len(fnames)-1))
        while True:
            random.shuffle(t0_idxs)
            for t0_idx in t0_idxs:
                for crops in parse_t0_t1(fnames[t0_idx], fnames[t0_idx+1],
                                         patch_size):
                    yield crops

    return (tf.data.Dataset.from_generator(crops, output_types=(tf.float32,
                                                                tf.float32,
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
    crops = parse_t0_t1(opts.fname_t0, opts.fname_t1,
                        patch_size=128, crops_per_img=4)
    rgb_t1, dither_t0, dither_t1 = next(crops)
    u.collage([
        u.rgb_img_to_pil(rgb_t1),
        u.dither_to_pil(dither_t0),
        u.dither_to_pil(dither_t1)
    ], side_by_side=True).show()
