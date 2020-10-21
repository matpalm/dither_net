#!/usr/bin/env python3

import argparse
from PIL import Image

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--rgb-base-dir', type=str, required=True)
parser.add_argument('--dither-base-dir', type=str, required=True)
parser.add_argument('--center-frame', type=int, required=True)
parser.add_argument('--frames-around', type=int, required=True)
opts = parser.parse_args()
print(opts)

w, h = 181, 134
imgs = []
for f in range(opts.center_frame - opts.frames_around,
               opts.center_frame + opts.frames_around):
    rgb_fname = "%s/f_%08d.jpg" % (opts.rgb_base_dir, f)
    rgb = Image.open(rgb_fname).resize((w, h))
    dither_fname = "%s/f_%08d.png" % (opts.dither_base_dir, f)
    dither = Image.open(dither_fname).resize((w, h))
    pair = Image.new('RGB', (w*2, h))
    pair.paste(rgb, (0, 0))
    pair.paste(dither, (w, 0))
    imgs.append(pair)

gif_fname = "f_%08d.gif" % opts.center_frame
print(gif_fname)
imgs[0].save(fp=gif_fname, format='GIF', append_images=imgs[1:],
             save_all=True, duration=40, loop=0, optimize=True)
