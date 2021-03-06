from objax.nn import Conv2D, Sequential, BatchNorm2D
import objax
from objax.nn.init import xavier_normal
from jax.nn.functions import gelu


class Discriminator(objax.Module):
    def __init__(self):
        self.model = Sequential(
            [Conv2D(1, 8, strides=2, k=5, use_bias=False),
             BatchNorm2D(8), gelu,
             Conv2D(8, 16, strides=2, k=3, use_bias=False),
             BatchNorm2D(16), gelu,
             Conv2D(16, 32, strides=2, k=3, use_bias=False),
             BatchNorm2D(32), gelu,
             Conv2D(32, 64, strides=2, k=3, use_bias=False),
             BatchNorm2D(64), gelu,
             Conv2D(64, 1, strides=1, k=1, use_bias=False)])  # logits

    def __call__(self, x, training):
        x = x.transpose((0, 3, 1, 2))
        return self.model(x, training=training)
