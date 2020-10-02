
import jax
import jax.numpy as jnp
import objax
from objax.nn.init import xavier_normal
from jax.nn.functions import gelu
from PIL import Image
import numpy as np
import util as u
from objax.nn import Conv2D, ConvTranspose2D


def _upsample_nearest_neighbour(inputs_nchw):
    # nearest neighbour upsampling on NCHW input
    _n, input_c, h, w = inputs_nchw.shape
    flat_inputs_shape = (-1, h, w, 1)
    flat_inputs = jnp.reshape(inputs_nchw, flat_inputs_shape)
    resize_kernel = jnp.ones((2, 2, 1, 1))
    strides = (2, 2)
    flat_outputs = jax.lax.conv_transpose(
        flat_inputs, resize_kernel, strides, padding="SAME")
    outputs_nchw_shape = (-1, input_c, 2 * h, 2 * w)
    outputs_nchw = jnp.reshape(flat_outputs, outputs_nchw_shape)
    return outputs_nchw


DEBUG = True


class Unet(objax.Module):
    def __init__(self):
        self.enc = Conv2D(3, 8, strides=2, k=5)
        self.dec = ConvTranspose2D(8, 8, strides=2, k=5)
        self.logits = Conv2D(8, 1, strides=1, k=1)

    def __call__(self, x, training):
        x = x.transpose((0, 3, 1, 2))
        if DEBUG:
            print("inp ", x.shape)

        y = gelu(self.enc(x))
        if DEBUG:
            print("enc ", y.shape)

        y = gelu(self.dec(y))
        if DEBUG:
            print("dec ", y.shape)

        logits = self.logits(y)
        if DEBUG:
            print("log ", logits.shape)
        return logits.transpose((0, 2, 3, 1))
