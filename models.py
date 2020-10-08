
import jax
import jax.numpy as jnp
import objax
from objax.nn.init import xavier_normal
from jax.nn.functions import gelu
from PIL import Image
import numpy as np
import util as u
from objax.nn import Conv2D, ConvTranspose2D, Sequential, BatchNorm2D


DEBUG = False


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


class EncoderBlock(objax.Module):

    def __init__(self, nin, nout, k):
        self.shortcut = Conv2D(nin, nout, strides=2, k=1)
        self.conv1 = Conv2D(nin, nout, strides=1, k=k)
        self.conv2 = Conv2D(nout, nout, strides=2, k=3)

    def __call__(self, x, training):
        if DEBUG:
            print(">x", x.shape)

        shortcut = self.shortcut(x)
        if DEBUG:
            print("shortcut", shortcut.shape)

        y = gelu(self.conv1(x))
        if DEBUG:
            print("c1", y.shape)
        y = gelu(self.conv2(y))
        if DEBUG:
            print("c2", y.shape)

        y += shortcut
        if DEBUG:
            print("y_shortcut", y.shape)

        if DEBUG:
            print("<y", y.shape)
        return y


class DecoderBlock(objax.Module):

    def __init__(self, nin, nout):
        self.shortcut = Conv2D(nin, nout, strides=1, k=1)
        self.conv1 = Conv2D(nin, nout, strides=1, k=3)
        self.conv2 = Conv2D(nout, nout, strides=1, k=3)
        self.skip_conv = Conv2D(2*nout, nout, strides=1, k=1)

    def __call__(self, x, encoded, training):
        if DEBUG:
            print(">x", x.shape)

        y = _upsample_nearest_neighbour(x)
        if DEBUG:
            print("up", y.shape)

        shortcut = self.shortcut(y)
        if DEBUG:
            print("shortcut", shortcut.shape)

        y = gelu(self.conv1(y))
        if DEBUG:
            print("c1", y.shape)

        y = gelu(self.conv2(y))
        if DEBUG:
            print("c2", y.shape)

        if encoded is not None:
            y = jnp.concatenate([y, encoded], axis=1)
            if DEBUG:
                print("skip_concat", y.shape)
            y = self.skip_conv(y)
            if DEBUG:
                print("skip_conv", y.shape)

        y += shortcut
        if DEBUG:
            print("y_shortcut", y.shape)

        if DEBUG:
            print("<y", y.shape)
        return y


class Generator(objax.Module):
    def __init__(self):

        num_channels = 3

        self.encoders = objax.ModuleList()
        k = 7
        for num_output_channels in [32, 64, 128]:  # 32, 64, 128, 128, 128]:
            self.encoders.append(EncoderBlock(
                num_channels, num_output_channels, k))
            k = 3
            num_channels = num_output_channels

        self.decoders = objax.ModuleList()
        for num_output_channels in [64, 32, 16]:  # , 128, 64, 32, 16]:
            self.decoders.append(DecoderBlock(
                num_channels, num_output_channels))
            num_channels = num_output_channels

        #self.logits_bn = BatchNorm2D(num_channels)
        self.logits = Conv2D(num_channels, nout=1,
                             strides=1, k=1, w_init=xavier_normal)

    def __call__(self, img, training=True):
        y = img.transpose((0, 3, 1, 2))
        if DEBUG:
            print("img", y.shape)

        encoded = []
        for e_idx, encoder in enumerate(self.encoders):
            if DEBUG:
                print(">e_%d" % e_idx)
            y = encoder(y, training)
            encoded.append(y)
            if DEBUG:
                print("<e_%d" % e_idx)

        for d_idx, decoder in enumerate(self.decoders):
            if DEBUG:
                print(">d_%d" % d_idx)
            enc = None
            if d_idx < len(self.decoders)-1:
                enc = encoded[-d_idx-2]
            y = decoder(y, enc, training)
            if DEBUG:
                print("<d_%d" % d_idx)

        #logits = self.logits(self.logits_bn(y, training))
        logits = self.logits(y)
        if DEBUG:
            print("l", logits.shape)
        return logits.transpose((0, 2, 3, 1))


class Discriminator(objax.Module):
    def __init__(self):
        self.model = Sequential(
            [Conv2D(1, 8, strides=2, k=5, use_bias=False),
             BatchNorm2D(8), gelu,
             Conv2D(8, 16, strides=2, k=3, use_bias=False),
             BatchNorm2D(16), gelu,
             Conv2D(16, 16, strides=2, k=3, use_bias=False),
             BatchNorm2D(16), gelu,
             Conv2D(16, 1, strides=1, k=1)])  # logits

    def __call__(self, x, training):
        x = x.transpose((0, 3, 1, 2))
        return self.model(x, training=training)
