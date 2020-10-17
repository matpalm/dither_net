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
                  help='file to save current frame in case of restart',
                  default='current_frame.txt')
args.add_argument('--time-between-frames', type=int, default=1,
                  help='time in sec between frames')
args.add_argument('--min-pixel-diff-for-refresh', type=int, default=100000,
                  help='trigger redraw if numbers of pixels changed > this')
args.add_argument('--max-redraw-rate', type=int, default=20,
                  help="don't redraw any more frequently that this")
args.add_argument('--use-virtual-display', action='store_true',
                  help='if set then use TK virtual EPD. works on desktop')
args.add_argument('--restart', action='store_true',
                  help='ignore current-frame-file and restart')
opts = args.parse_args()


def log(s):
    datetimestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s: %s" % (datetimestamp, s))
    sys.stdout.flush()


log("opts %s" % opts)

# slurp manifest
frames = list(map(str.strip, open(opts.manifest, 'r').readlines()))
log("%d frames read from manifest %s" % (len(frames), opts.manifest))

# decide which frame to start
if opts.restart:
    # forced restart
    frame_idx = 0
elif os.path.exists(opts.current_frame_file):
    # check to see if we are in progress
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

# first call should use slower/flashier full update but from then on we can
# direct update to display just differences when possible, but more frequently
# than once every N frames
first_display = True
last_redraw_np = None
frames_since_last_redraw = 0

# init display
if opts.use_virtual_display:
    display = VirtualEPDDisplay(dims=(1448, 1072))  # , rotate='flip')
else:
    display = AutoEPDDisplay(vcom=-2.25, spi_hz=24000000)

while frame_idx < len(frames):

    fname = frames[frame_idx]

    # load dither and paste into frame buffer
    dither = Image.open(fname)
    dither_np = np.array(dither)
    display.frame_buf.paste(dither, (0, 0))

    pixel_diff = None
    if first_display:
        do_full_redraw = True
        first_display = False
    else:
        # always calc pixel_diff for debugging. note: we actually just care
        # about pixels that have been white and are now black, that's where we
        # see the ghosting, not just pixels that differ.
        pixel_diff = np.sum(last_redraw_np != dither_np)
        if frames_since_last_redraw < opts.max_redraw_rate:
            do_full_redraw = False
        else:
            do_full_redraw = pixel_diff > opts.min_pixel_diff_for_refresh

    if do_full_redraw:
        display.draw_full(constants.DisplayModes.GC16)
        last_redraw_np = dither_np
        frames_since_last_redraw = 0
    else:
        display.draw_partial(constants.DisplayModes.DU4)

    # write progress in case we get killed
    with open(opts.current_frame_file, 'w') as f:
        print(fname, file=f)

    # log, inc frame count and sleep a bit
    log("%s %s %s %s" % (fname, frames_since_last_redraw, pixel_diff,
                         do_full_redraw))
    frame_idx += 1
    frames_since_last_redraw += 1
    time.sleep(opts.time_between_frames)


# if we get through then remove progress file so we
# can start again
log("done, restarting")
os.remove(opts.current_frame_file)
