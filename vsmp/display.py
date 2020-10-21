from IT8951.display import AutoEPDDisplay
import sys
from PIL import Image
from IT8951 import constants

display = AutoEPDDisplay(vcom=-2.25, rotate=None, spi_hz=24000000)
dither = Image.open(sys.argv[1])
display.frame_buf.paste(dither, (0, 0))
display.draw_full(constants.DisplayModes.GC16)
