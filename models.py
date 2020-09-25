
import jax
import jax.numpy as jnp
import objax
from objax.variable import TrainVar
from jax.nn.initializers import glorot_normal, he_normal
from jax.nn.functions import gelu
from PIL import Image
import numpy as np


def _conv_layer(stride, activation, kernel_size, inp, kernel, bias):
    no_dilation = (1, 1)
    some_height_width = 10  # values don't matter; just shape of input
    input_shape = (1, some_height_width, some_height_width, 3)
    kernel_shape = (kernel_size, kernel_size, 1, 1)
    input_kernel_output = ('NHWC', 'HWIO', 'NHWC')
    conv_dimension_numbers = jax.lax.conv_dimension_numbers(input_shape,
                                                            kernel_shape,
                                                            input_kernel_output)
    block = jax.lax.conv_general_dilated(inp, kernel, (stride, stride),
                                         'SAME', no_dilation, no_dilation,
                                         conv_dimension_numbers)
    if bias is not None:
        block += bias
    if activation:
        block = activation(block)
    return block


def _upsample_nearest_neighbour(inputs):
    input_channels = inputs.shape[-1]
    inputs_nchw = jnp.transpose(inputs, (0, 3, 1, 2))
    flat_inputs_shape = (-1, inputs.shape[1], inputs.shape[2], 1)
    flat_inputs = jnp.reshape(inputs_nchw, flat_inputs_shape)

    resize_kernel = jnp.ones((2, 2, 1, 1))
    strides = (2, 2)
    flat_outputs = jax.lax.conv_transpose(
        flat_inputs, resize_kernel, strides, padding="SAME")

    outputs_nchw_shape = (-1, input_channels, 2 *
                          inputs.shape[1], 2 * inputs.shape[2])
    outputs_nchw = jnp.reshape(flat_outputs, outputs_nchw_shape)
    return jnp.transpose(outputs_nchw, (0, 2, 3, 1))


class Unet(objax.Module):
    def __init__(self):

        key = objax.random.Generator(123)

        self.enc_conv_kernels = objax.ModuleList()
        self.enc_conv_biases = objax.ModuleList()
        num_channels = 3
        # TODO: drop i here
        for i, num_output_channels in enumerate([32, 64, 128, 256, 256]):
            self.enc_conv_kernels.append(TrainVar(
                he_normal()(key(), (3, 3, num_channels, num_output_channels))))
            self.enc_conv_biases.append(
                TrainVar(jnp.zeros((num_output_channels,))))
            num_channels = num_output_channels

        self.dec_conv_kernels = objax.ModuleList()
        self.dec_conv_biases = objax.ModuleList()
        for i, num_output_channels in enumerate([256, 128, 64, 32, 16]):
            self.dec_conv_kernels.append(TrainVar(
                he_normal()(key(), (3, 3, num_channels, num_output_channels))))
            self.dec_conv_biases.append(
                TrainVar(jnp.zeros((num_output_channels,))))
            num_channels = num_output_channels

        self.dec_skip_conv_kernels = objax.ModuleList()
        self.dec_skip_conv_biases = objax.ModuleList()
        for channels in [256, 128, 64, 32]:
            self.dec_skip_conv_kernels.append(TrainVar(
                he_normal()(key(), (3, 3, 2*channels, channels))))
            self.dec_skip_conv_biases.append(
                TrainVar(jnp.zeros((channels,))))

        self.logits_conv_kernel = TrainVar(
            glorot_normal()(key(), (1, 1, num_channels, 1)))
        self.logits_conv_bias = TrainVar(jnp.zeros((1,)))

    def dither_logits(self, img):
        y = img
        # print("inp", y.shape)

        encoded = []
        for e in range(5):
            if e == 0:
                kernel_size = 7
            else:
                kernel_size = 3
            y = _conv_layer(2, gelu, kernel_size, y,
                            self.enc_conv_kernels[e].value,
                            self.enc_conv_biases[e].value)
            encoded.append(y)
            # print(y.shape)

        for d in range(5):

            upsampled = _upsample_nearest_neighbour(y)

            y = _conv_layer(1, gelu, 3, upsampled,
                            self.dec_conv_kernels[d].value,
                            self.dec_conv_biases[d].value)

            if d < len(self.dec_conv_kernels)-1:
                # print("d", d, y.shape)
                y = jnp.concatenate([y, encoded[3-d]], axis=3)
                # print("d+e", d, y.shape)
                y = _conv_layer(1, gelu, 3, y,
                                self.dec_skip_conv_kernels[d].value,
                                self.dec_skip_conv_biases[d].value)
                # print("d+e_2", d, y.shape)
            # else:
            #     print("d", d, y.shape)

        logits = _conv_layer(1, None, 1, y,
                             self.logits_conv_kernel.value,
                             self.logits_conv_bias.value)
        # print(logits.shape)

        return logits

    def dither_output(self, img):
        pred_dither = self.dither_logits(img)
        lit_pixels = pred_dither[0, :, :, 0] > 0
        lit_pixels = jnp.where(lit_pixels, 255, 0).astype(jnp.uint8)
        return Image.fromarray(np.array(lit_pixels), 'L')
