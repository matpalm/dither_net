import time
import datetime
import numpy as np
from PIL import Image
import os
import math
import jax.numpy as jnp
from collections import namedtuple
import re


def DTS():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def rgb_img_to_pil(rgb_img):
    rgb_img = np.array(rgb_img * 255, dtype=np.uint8)
    return Image.fromarray(rgb_img)


def dither_to_pil(dither, threshold=0):
    lit_pixels = dither[:, :, 0] > threshold
    lit_pixels = np.where(lit_pixels, 255, 0).astype(np.uint8)
    return Image.fromarray(np.array(lit_pixels), 'L')


def collage(pil_imgs, side_by_side=False):
    # assume all imgs same (w, h)
    w, h = pil_imgs[0].size
    if side_by_side:
        collage = Image.new('RGB', (len(pil_imgs)*w, h))
        for idx, img in enumerate(pil_imgs):
            collage.paste(img, (idx*w, 0))
    else:
        n = math.ceil(math.sqrt(len(pil_imgs)))
        collage = Image.new('RGB', (n*w, n*h))
        for idx, img in enumerate(pil_imgs):
            r, c = idx % n, idx // n
            collage.paste(img, (r*w, c*h))
    return collage


class ValueFromFile(object):
    def __init__(self, fname, init_value):
        self.current_value = init_value
        self.fname = fname

    def value(self):
        try:
            self.current_value = float(open(self.fname).read())
        except Exception as e:
            print("couldn't reload value " + str(e))
        return self.current_value


def clip_gradients(grads, theta):
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
    scale_factor = jnp.minimum(theta / total_grad_norm, 1.)
    return [g * scale_factor for g in grads]


def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))


def frame_num(fname):
    m = re.match(".*f_(\d*)\.*", fname)
    if m:
        return int(m.group(1))
    else:
        raise Exception("no frame num in fname [%s]" % fname)


# ManifestEntry = namedtuple('ManifestEntry', 'fname scene_change')


# def read_manifest(manifest_file, has_scene_change):
#     manifest = []
#     for line in map(str.strip, open(manifest_file, 'r').readlines()):
#         if has_scene_change:

#             fname, scene_change = line.split(" ")
#             manifest.append(ManifestEntry(fname, eval(scene_change)))
#         else:
#             manifest.append(line)
#     return manifest
