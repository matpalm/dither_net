
import jax
import jax.numpy as jnp
import objax
#from objax.variable import TrainVar
from objax.nn.init import xavier_normal
from jax.nn.functions import gelu
from PIL import Image
import numpy as np
import util as u
from objax.nn import Conv2D, BatchNorm2D


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


class Unet(objax.Module):
    def __init__(self):

        num_channels = 3

        self.enc_bn = objax.ModuleList()
        self.enc_conv = objax.ModuleList()
        k = 7
        for num_output_channels in [32, 64, 128, 256, 256]:
            self.enc_bn.append(BatchNorm2D(num_channels))
            self.enc_conv.append(Conv2D(num_channels, num_output_channels,
                                        strides=2, k=k))
            k = 3
            num_channels = num_output_channels

        self.dec_bn = objax.ModuleList()
        self.dec_conv = objax.ModuleList()
        for num_output_channels in [256, 128, 64, 32, 16]:
            self.dec_bn.append(BatchNorm2D(num_channels))
            self.dec_conv.append(Conv2D(num_channels, num_output_channels,
                                        strides=1, k=3))
            num_channels = num_output_channels

        self.skip_dec_bn = objax.ModuleList()
        self.skip_dec_conv = objax.ModuleList()
        for channels in [256, 128, 64, 32]:
            self.skip_dec_bn.append(BatchNorm2D(2*channels))
            self.skip_dec_conv.append(Conv2D(2*channels, channels,
                                             strides=1, k=1))
            num_channels = num_output_channels

        self.logits_bn = BatchNorm2D(num_channels)
        self.logits = Conv2D(num_channels, nout=1, strides=1, k=1,
                             w_init=xavier_normal)

    def __call__(self, img, training):
        debug = False

        y = img.transpose((0, 3, 1, 2))

        if debug:
            print("img", y.shape)

        encoded = []
        for e_idx, (bn, conv) in enumerate(zip(self.enc_bn, self.enc_conv)):
            y = gelu(conv(bn(y, training)))
            encoded.append(y)
            if debug:
                print("e_%d" % e_idx, y.shape)

        for d_idx, (bn, conv) in enumerate(zip(self.dec_bn, self.dec_conv)):
            y = _upsample_nearest_neighbour(y)
            if debug:
                print("up", y.shape)

            y = gelu(conv(bn(y, training)))
            if debug:
                print("d_%d" % d_idx, y.shape)

            if d_idx < len(self.skip_dec_conv):
                y = jnp.concatenate([y, encoded[3-d_idx]], axis=1)
                if debug:
                    print("d+e_%d" % d_idx, y.shape)
                bn, conv = self.skip_dec_bn[d_idx], self.skip_dec_conv[d_idx]
                y = gelu(conv(bn(y, training)))
                if debug:
                    print("d+e_%d conv" % d_idx, y.shape)

        logits = self.logits(self.logits_bn(y, training))
        if debug:
            print("l", logits.shape)
        #print("return", logits.transpose((0, 2, 3, 1)).shape)
        return logits.transpose((0, 2, 3, 1))

    def dithers_as_pil(self, imgs):
        return [u.dither_to_pil_image(d) for d in self.__call__(imgs, training=False)]


if __name__ == '__main__':
    unet = Unet()
    import numpy as np
    print(unet(np.random(4, 128, 128, 3)).shape)
