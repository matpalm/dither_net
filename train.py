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
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--gradient-clip', type=float, default=1.0)
parser.add_argument('--steps-per-epoch', type=int)
parser.add_argument('--positive-weight', type=float, default=1.0)
opts = parser.parse_args()
print(opts)

RUN = u.DTS()

wandb.init(project='dither_net', group='v1', name=RUN)

unet = models.Unet()
print(unet.vars())


def _cross_entropy(rgb_img, true_dither, positive_weight, training):
    pred_dither_logits = unet(rgb_img, training)
    per_pixel_loss = sigmoid_cross_entropy_logits(pred_dither_logits,
                                                  true_dither)
    loss_weight = jnp.where(true_dither == 1, positive_weight, 1.0)
    return jnp.mean(loss_weight * per_pixel_loss)


def cross_entropy_non_training(rgb_img, true_dither):
    return _cross_entropy(rgb_img, true_dither,
                          positive_weight=opts.positive_weight, training=False)


cross_entropy_non_training = objax.Jit(cross_entropy_non_training, unet.vars())


def cross_entropy_training(rgb_img, true_dither):
    return _cross_entropy(rgb_img, true_dither,
                          positive_weight=opts.positive_weight, training=True)


gradient_loss = objax.GradValues(cross_entropy_training, unet.vars())

optimiser = objax.optimizer.Adam(unet.vars())
#optimiser = objax.optimizer.Momentum(unet.vars(), momentum=0.9, nesterov=True)


def train_step(learning_rate, rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, theta=opts.gradient_clip)
    optimiser(learning_rate, grads)


def clip_gradients(grads, theta):
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
    scale_factor = jnp.minimum(theta / total_grad_norm, 1.)
    return [g * scale_factor for g in grads]


def train_step_with_grad_norms(learning_rate, rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, theta=opts.gradient_clip)
    optimiser(learning_rate, grads)
    grad_norms = [jnp.linalg.norm(g) for g in grads]
    return grad_norms


train_step = objax.Jit(train_step,
                       gradient_loss.vars() + optimiser.vars())

train_step_with_grad_norms = objax.Jit(train_step_with_grad_norms,
                                       gradient_loss.vars() + optimiser.vars())


def predict(rgb_imgs):
    return unet(rgb_imgs, training=False)


predict = objax.Jit(predict, unet.vars())


learning_rate = u.ValueFromFile("learning_rate.txt", 1e-3)
# improvement_tracking = u.ImprovementTracking(patience=10, smoothing=0.5)

# for debug images
u.ensure_dir_exists("test/%s" % RUN)
if os.path.exists("test/latest"):
    os.remove("test/latest")
os.symlink(RUN, "test/latest")

# load some full res images
full_rgbs = []
full_true_dithers = []
for frame in [5000, 43000, 67000, 77000]:
    full_rgb, full_true_dither = data.parse_full_size(
        "frames/full_res/f_%08d.jpg" % frame)
    full_rgbs.append(full_rgb)
    full_true_dithers.append(full_true_dither)
full_rgbs = np.stack(full_rgbs)
full_true_dithers = np.stack(full_true_dithers)

dataset = data.dataset(manifest_file=opts.manifest_file,
                       batch_size=opts.batch_size)

ckpt = objax.io.Checkpoint(logdir=f"ckpts/{RUN}/", keep_ckpts=20)

for epoch in range(10000):

    g_min_max = None

    for rgb_imgs, true_dithers in dataset.take(opts.steps_per_epoch):
        rgb_imgs = rgb_imgs.numpy()
        true_dithers = true_dithers.numpy()

        # collect grad norms once per epoch
        if g_min_max is None:
            grad_norms = train_step_with_grad_norms(
                learning_rate.value(), rgb_imgs, true_dithers)
            g_min_max = (float(jnp.min(grad_norms)),
                         float(jnp.max(grad_norms)))
        else:
            train_step(learning_rate.value(), rgb_imgs, true_dithers)

    # ckpt
    ckpt.save(unet.vars(), idx=epoch)

    # check loss against last batch as well as sample batch
    # TODO: jit these?
    last_loss = float(cross_entropy_non_training(rgb_imgs, true_dithers))

    # save sample rgb, true dith and pred dither in collage.
    full_pred_dithers = predict(full_rgbs)
    samples = []
    for r, t, p in zip(full_rgbs, full_true_dithers, full_pred_dithers):
        triplet = [u.rgb_img_to_pil(r), u.dither_to_pil(t), u.dither_to_pil(p)]
        samples.append(u.collage(triplet, side_by_side=True))
    u.collage(samples).save("test/%s/%05d.png" % (RUN, epoch))

    # and prep images for wandb logging
    # wand_imgs = []
    # for d, f in zip(sample_dithered_img, FRAMES):
    #     wand_imgs.append(wandb.Image(d, caption="f_%05d" % f))
    wandb.log({'loss': last_loss, 'g_min': g_min_max[0], 'g_max': g_min_max[1],
               'learning_rate': learning_rate.value()}, step=epoch)

    # if not improvement_tracking.improved(loss):
    #     learning_rate /= 2
    #     improvement_tracking.reset()

    print("epoch", epoch, "lr", learning_rate.value(),
          "g_min max", g_min_max, "last_loss", last_loss)
