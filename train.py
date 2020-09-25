import objax
import models
import jax.numpy as jnp
from objax.functional.loss import sigmoid_cross_entropy_logits
from PIL import Image
import numpy as np
import util as u
import wandb


wandb.init(project='dither_net', group='v1', name=u.DTS())


F = 72000

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
    pred_dither_logits = unet.dither_logits(rgb_img)
    per_pixel_loss = sigmoid_cross_entropy_logits(pred_dither_logits,
                                                  true_dither)
    return jnp.mean(per_pixel_loss)


gradient_loss = objax.GradValues(cross_entropy, unet.vars())

optimiser = objax.optimizer.Adam(unet.vars())


def train_step(learning_rate, rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    optimiser(learning_rate, grads)


train_step = objax.Jit(train_step,
                       gradient_loss.vars() + optimiser.vars())

learning_rate = 1e-3
improvement_tracking = u.ImprovementTracking(smoothing=0.1)

for i in range(100):
    print(">", i)

    for _ in range(100):
        train_step(learning_rate, rgb_img, true_dither)

    loss = float(cross_entropy(rgb_img, true_dither))
    sample_dithered_img = unet.dither_output(rgb_img)

    wandb.log({'loss': loss,
               'learning_rate': learning_rate,
               'eg_img': [wandb.Image(sample_dithered_img, caption="Label")]},
              step=i)

    if not improvement_tracking.improved(loss):
        learning_rate /= 2
        improvement_tracking.reset()

    print(i, learning_rate, loss)
