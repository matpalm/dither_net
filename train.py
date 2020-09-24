import objax
import models
import jax.numpy as jnp
from objax.functional.loss import sigmoid_cross_entropy_logits
from PIL import Image
import numpy as np

F = 25000

rgb_img = Image.open("imgs/c/c_%07d.png" % F).resize((1440, 1056))
true_dither = rgb_img.convert(mode='1', dither=Image.FLOYDSTEINBERG)

rgb_img = np.array(rgb_img, dtype=np.float32)
rgb_img /= 255.
rgb_img = np.expand_dims(rgb_img, 0)  # single elem batch

true_dither = np.array(true_dither, dtype=np.float32)
true_dither = np.expand_dims(true_dither, -1)  # single channel
true_dither = np.expand_dims(true_dither, 0)   # single elem batch

unet = models.Unet()


def cross_entropy(rgb_img, true_dither):
    pred_dither_logits = unet.dither(rgb_img)
    per_pixel_loss = sigmoid_cross_entropy_logits(pred_dither_logits,
                                                  true_dither)
    return jnp.mean(per_pixel_loss)


gradient_loss = objax.GradValues(cross_entropy, unet.vars())

optimiser = objax.optimizer.Adam(unet.vars())
learning_rate = 1e-4


def train_step(rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    optimiser(learning_rate, grads)


train_step = objax.Jit(train_step,
                       gradient_loss.vars() + optimiser.vars())


def dithered_output():
    pred_dither = unet.dither(rgb_img)
    lit_pixels = pred_dither[0, :, :, 0] > 0
    return Image.fromarray(np.where(lit_pixels, 255, 0).astype(np.uint8), 'L')


dithered_output().save("test.png")
