import objax
import models
import jax.numpy as jnp
from objax.functional.loss import sigmoid_cross_entropy_logits
from PIL import Image
import numpy as np
import util as u
import wandb
import data
import argparse


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
parser.add_argument('--batch-size', type=int)
opts = parser.parse_args()
print(opts)

RUN = u.DTS()

wandb.init(project='dither_net', group='v1', name=RUN)

unet = models.Unet()


def cross_entropy(rgb_img, true_dither):
    pred_dither_logits = unet.dither_logits(rgb_img)
    per_pixel_loss = sigmoid_cross_entropy_logits(pred_dither_logits,
                                                  true_dither)
    return jnp.mean(per_pixel_loss)


gradient_loss = objax.GradValues(cross_entropy, unet.vars())

optimiser = objax.optimizer.Adam(unet.vars())
# optimiser = objax.optimizer.Momentum(unet.vars(), momentum=0.1, nesterov=True)


def train_step(learning_rate, rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, 1)
    optimiser(learning_rate, grads)


def clip_gradients(grads, theta):
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
    scale_factor = jnp.minimum(theta / total_grad_norm, 1.)
    return [g * scale_factor for g in grads]


def train_step_with_grad_norms(learning_rate, rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, theta=1)
    optimiser(learning_rate, grads)
    grad_norms = [jnp.linalg.norm(g) for g in grads]
    return grad_norms


train_step = objax.Jit(train_step,
                       gradient_loss.vars() + optimiser.vars())

train_step_with_grad_norms = objax.Jit(train_step_with_grad_norms,
                                       gradient_loss.vars() + optimiser.vars())

learning_rate = 1e-3
# improvement_tracking = u.ImprovementTracking(patience=10, smoothing=0.5)

# for debug images
u.ensure_dir_exists("test/%s" % RUN)

sample_batch_rgb = None

for i in range(10000):

    g_min_max = None
    for _ in range(100):
        dataset = data.dataset(manifest_file=opts.manifest_file,
                               batch_size=opts.batch_size)
        for rgb_imgs, true_dithers in dataset:
            rgb_imgs = rgb_imgs.numpy()
            true_dithers = true_dithers.numpy()
            # grab first batch as a sample batch for use over entire training
            if sample_batch_rgb is None:
                sample_batch_rgb = rgb_imgs
            # collect grad norms for first step, once per epoch
            if g_min_max is None:
                grad_norms = train_step_with_grad_norms(
                    learning_rate, rgb_imgs, true_dithers)
                g_min_max = (float(jnp.min(grad_norms)),
                             float(jnp.max(grad_norms)))
            else:
                train_step(learning_rate, rgb_imgs, true_dithers)

    # check loss against last batch
    loss = float(cross_entropy(rgb_imgs, true_dithers))

    # save dithers to disk
    sample_dithered_img = unet.dithers_as_pil(sample_batch_rgb)
    u.collage(sample_dithered_img).save("test/%s/%05d.png" % (RUN, i))

    # and prep images for wandb logging
    # wand_imgs = []
    # for d, f in zip(sample_dithered_img, FRAMES):
    #     wand_imgs.append(wandb.Image(d, caption="f_%05d" % f))
    wandb.log({'loss': loss, 'g_min': g_min_max[0], 'g_max': g_min_max[1],
               'learning_rate': learning_rate}, step=i)

    # if not improvement_tracking.improved(loss):
    #     learning_rate /= 2
    #     improvement_tracking.reset()

    print("i", i, "lr", learning_rate, "g_min max", g_min_max, "loss", loss)
