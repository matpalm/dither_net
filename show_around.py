#!/usr/bin/env python3

import sys
import util as u
from PIL import Image

D = "frames/20201008_201718/"
fname = sys.argv[1]
frame_num = u.frame_num_of(fname)


def frame_num_to_name(frame_num):
    return "f_%08d.png" % frame_num


imgs = []
for i in range(frame_num-2, frame_num+3):
    fname = "%s/f_%08d.png" % (D, i)
    print(fname)
    img = Image.open(fname)
    imgs.append(img)

u.collage(imgs, side_by_side=True).resize((2000, 300)).show()
