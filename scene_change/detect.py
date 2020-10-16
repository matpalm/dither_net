#!/usr/bin/env python3

from PIL import Image
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
opts = parser.parse_args()


class StdScoreAnomalyDetector(object):
    # simpler helper to estimate scene changes

    def __init__(self, window_size=50, anomaly_z_threshold=5):
        self.window = []
        self.window_size = window_size
        self.anomaly_z_threshold = anomaly_z_threshold

    def anomaly(self, v):
        self.window.append(v)
        if len(self.window) <= self.window_size+1:
            return False
        self.window.pop(0)
        mean = np.mean(self.window)
        std = np.std(self.window)
        z_score = abs(mean - v) / std
        if z_score > self.anomaly_z_threshold:
            self.window = []
            return True
        return False


pixel_same_count = StdScoreAnomalyDetector()
fnames = sorted(map(str.strip, open(opts.manifest_file).readlines()))

last_dither_np = None
for fname in fnames:
    rgb_img = Image.open(fname)
    dither = rgb_img.convert(mode='1', dither=Image.FLOYDSTEINBERG)
    dither_np = np.array(dither)

    # compare to last dither in terms of number of same pixels
    # decide if it looks anomalious enough to be a scene change
    if last_dither_np is None:
        scene_change = True
    else:
        count = np.sum(dither_np == last_dither_np)
        scene_change = pixel_same_count.anomaly(count)
    last_dither_np = dither_np

    print(fname, scene_change)
