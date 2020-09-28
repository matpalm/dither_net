
import jax
import jax.numpy as jnp
import objax
from objax.variable import TrainVar
from jax.nn.initializers import glorot_normal, he_normal
from jax.nn.functions import gelu
from PIL import Image
import numpy as np
import util as u


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

        self.encoders = objax.ModuleList()
        k = 7
        for num_output_channels in [32, 64, 128, 256, 512]:
            self.encoders.append(objax.nn.Conv2D(
                num_channels, num_output_channels, strides=2, k=k))
            k = 3
            num_channels = num_output_channels

        self.decoders = objax.ModuleList()
        for num_output_channels in [256, 128, 64, 32, 16]:
            self.decoders.append(objax.nn.Conv2D(
                num_channels, num_output_channels, strides=1, k=3))
            num_channels = num_output_channels

        self.skip_decoders = objax.ModuleList()
        for channels in [256, 128, 64, 32]:
            self.skip_decoders.append(objax.nn.Conv2D(
                2*channels, channels, strides=1, k=1))
            num_channels = num_output_channels

        self.logits = objax.nn.Conv2D(num_channels, nout=1, strides=1, k=1)

    def __call__(self, img, training):
        y = img.transpose((0, 3, 1, 2))
        print_sizes = False
        if print_sizes:
            print("img", y.shape)

        encoded = []
        for e_idx, encoder in enumerate(self.encoders):
            y = gelu(encoder(y))
            encoded.append(y)
            if print_sizes:
                print("e_%d" % e_idx, y.shape)

        for d_idx in range(len(self.decoders)):
            y = _upsample_nearest_neighbour(y)
            if print_sizes:
                print("up", y.shape)

            y = gelu(self.decoders[d_idx](y))
            if print_sizes:
                print("d_%d" % d_idx, y.shape)

            if d_idx < len(self.skip_decoders):
                y = jnp.concatenate([y, encoded[3-d_idx]], axis=1)
                if print_sizes:
                    print("d+e_%d" % d_idx, y.shape)
                y = gelu(self.skip_decoders[d_idx](y))
                if print_sizes:
                    print("d+e_%d conv" % d_idx, y.shape)

        logits = self.logits(y)
        if print_sizes:
            print("l", logits.shape)
        return logits.transpose((0, 2, 3, 1))

    def dithers_as_pil(self, imgs):
        return [u.dither_to_pil_image(d) for d in self.__call__(imgs, training=False)]


if __name__ == '__main__':
    unet = Unet()
    import numpy as np
    print(unet(np.random(4, 128, 128, 3)).shape)
