import argparse

from IT8951.display import AutoEPDDisplay, VirtualEPDDisplay
from IT8951 import constants
from PIL import Image
import os
import time
import numpy as np
import datetime
import sys

args = argparse.ArgumentParser()
args.add_argument('--manifest', type=str,
                  help='list of all files to display',
                  default='manifest.txt')
args.add_argument('--current-frame-file', type=str,
                  help='where to save current frame being displayed in case of restart',
                  default='current_frame.txt')
args.add_argument('--time-between-frames', type=int, default=1,
                  help='time in sec between frames')
args.add_argument('--use-virtual-display', action='store_true',
                  help='if set then use TK virtual EPD. works on desktop')
opts = args.parse_args()


def log(s):
    datetimestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s: %s" % (datetimestamp, s))
    sys.stdout.flush()


log("opts %s" % opts)

# slurp manifest
frames = list(map(str.strip, open(opts.manifest, 'r').readlines()))
log("%d frames read from manifest %s" % (len(frames), opts.manifest))

# check to see if we are in progress
if os.path.exists(opts.current_frame_file):
    current_frame = open(opts.current_frame_file).read().strip()
    try:
        frame_idx = frames.index(current_frame) + 1
        log("skip to frame_idx %d" % frame_idx)
    except ValueError:
        raise Exception("current frame [%s] not in manifest [%s] (??)" % (
            current_frame, opts.manifest))
else:
    # otherwise dft to first frame
    frame_idx = 0

# first call should use slower/flashier full update
# but from then on we can direct update to display just
# differences
first_display = True

# init display
if opts.use_virtual_display:
    display = VirtualEPDDisplay(dims=(1448, 1072))  # , rotate='flip')
else:
    display = AutoEPDDisplay(vcom=-2.25, spi_hz=24000000)

# simpler helper to know when to occasionally force a flashy reset


class StdScoreAnomaly(object):

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


pixel_count = StdScoreAnomaly()
last_dither_np = None
full_redraw = True
same_pixel_count = None

while frame_idx < len(frames):

    fname = frames[frame_idx]

    # load dither and paste into frame buffer
    dither = Image.open(fname)
    display.frame_buf.paste(dither, (0, 0))

    # check current frame to last frame and count #pixels that
    # are the same. if this has shifted a lot declare it a new
    # scene that requires a full redraw
    dither_np = np.array(dither)
    if last_dither_np is not None:
        same_pixel_count = np.sum(dither_np == last_dither_np)
        full_redraw = pixel_count.anomaly(same_pixel_count)
    last_dither_np = dither_np

    # if doing a full redraw then use full reset, otherwise
    # do partial update (that still sadly results in ghosting)
    if full_redraw:
        display.draw_full(constants.DisplayModes.GC16)
    else:
        display.draw_partial(constants.DisplayModes.DU4)

    # write progress in case we get killed
    with open(opts.current_frame_file, 'w') as f:
        print(fname, file=f)

    # log, inc frame count and sleep a bit
    log("%s %s %s" % (fname, full_redraw, same_pixel_count))
    frame_idx += 1
    time.sleep(opts.time_between_frames)

# if we get through then remove progress file so we
# can start again
log("done, restarting")
os.remove(opts.current_frame_file)
