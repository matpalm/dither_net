import time
import datetime
import numpy as np
from PIL import Image
import os


def DTS():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def dither_to_pil_image(dither):
    lit_pixels = dither[:, :, 0] > 0
    lit_pixels = np.where(lit_pixels, 255, 0).astype(np.uint8)
    return Image.fromarray(np.array(lit_pixels), 'L')


class ImprovementTracking(object):
    def __init__(self, patience=3, burn_in=5, max_runtime=None,
                 smoothing=0.0):
        # smoothing = 0.0 => no smoothing

        self.original_patience = patience
        self.original_burn_in = burn_in
        self.max_runtime = max_runtime

        if smoothing < 0.0 or smoothing > 1.0:
            raise Exception("invalid smoothing value %s" % smoothing)
        self.smoothing = 1.0 - smoothing

        self.reset()

    def reset(self):
        self.patience = self.original_patience
        self.burn_in = self.original_burn_in
        self.lowest_value = None

        if self.max_runtime is not None:
            self.exit_time = time.time() + self.max_runtime
        else:
            self.exit_time = None

        self.smoothed_value = None

    def improved(self, value):
        # calc smoothed value
        if self.smoothed_value is None:
            print("IT first value is value")
            self.smoothed_value = value
        else:
            self.smoothed_value += self.smoothing * \
                (value - self.smoothed_value)
            print("IT value", value, "smoothed to", self.smoothed_value)

        # run taken too long?
        if self.exit_time is not None:
            print("IT timeout")
            if time.time() > self.exit_time:
                return False

        # ignore first burn_in iterations
        if self.burn_in > 0:
            print("IT still burning in", self.burn_in)
            self.burn_in -= 1
            return True

        # take very first value we see as the lowest
        if self.lowest_value is None:
            print("IT first value is lowest", self.smoothed_value)
            self.lowest_value = self.smoothed_value

        # check if we've made an improvement; if so reset patience and record
        # new lowest
        made_improvement = self.smoothed_value < self.lowest_value
        if made_improvement:
            self.patience = self.original_patience
            self.lowest_value = self.smoothed_value
            print("IT made improvement, reset patience, new lowest",
                  self.lowest_value)
            return True

        # if no improvement made reduce patience. if no more patience exit.
        self.patience -= 1
        print("IT didn't make improvement, patience now", self.patience)
        return self.patience != 0
