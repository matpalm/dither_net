import objax
import generator as g
import discriminator as d
import jax.numpy as jnp
from objax.functional import sigmoid
#from objax.functional.loss import sigmoid_cross_entropy_logits, mean_absolute_error
from PIL import Image
import numpy as np
import util as u
import wandb
import data
import argparse
import os
import time

JIT = True

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--gradient-clip', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--max-run-time', type=int, default=None,
                    help='max run time in secs')
parser.add_argument('--steps-per-epoch', type=int)
parser.add_argument('--positive-weight', type=float, default=1.0)
parser.add_argument('--reconstruction-loss-weight', type=float, default=1.0)
parser.add_argument('--discriminator-loss-weight', type=float, default=1.0)
parser.add_argument('--generator-sigmoid-b', type=float, default=1.0)
parser.add_argument('--discriminator-weight-clip', type=float, default=0.1)

opts = parser.parse_args()
print(opts)

finish_time = None
if opts.max_run_time is not None and opts.max_run_time > 0:
    finish_time = time.time() + opts.max_run_time

RUN = u.DTS()

wandb.init(project='dither_net', group='v1', name=RUN)

generator = g.Generator()
discriminator = d.Discriminator()

print("generator", generator.vars())
print("discriminator", discriminator.vars())


def steep_sigmoid(x):
    # since D from true_dithers only sees (0, 1) we want to make it's
    # job harder for fake_dithers by squashing the sigmoid activation towards
    # 0 & 1. simplest way to do this is by varying B in the generalised
    # logisitic function. this is handy since that makes it just a function on
    # x, so we can continue to use the numerically stable jax.special.expit
    # that objax.sigmoid wraps.
    # see https://en.wikipedia.org/wiki/Generalised_logistic_function
    return sigmoid(opts.generator_sigmoid_b * x)


def generator_loss(rgb_img, true_dither):
    # generator loss is based on the generated images from the RGB and is
    # composed of two components...
    pred_dither = steep_sigmoid(generator(rgb_img))

    # 1) a comparison to the true_dither to see how well it reconstructs it
    per_pixel_reconstruction_loss = jnp.abs(pred_dither - true_dither)
    loss_weight = jnp.where(true_dither == 1, opts.positive_weight, 1.0)
    reconstruction_loss = jnp.mean(loss_weight * per_pixel_reconstruction_loss)

    # 2) how well it fools the discriminator
    discriminator_logits = discriminator(pred_dither, training=False)
    overall_patch_loss = -jnp.mean(discriminator_logits)

    # overall loss is weighted combination of the two
    overall_loss = (reconstruction_loss * opts.reconstruction_loss_weight +
                    overall_patch_loss * opts.discriminator_loss_weight)

    return (overall_loss,
            {'reconstruction_loss':
             reconstruction_loss * opts.reconstruction_loss_weight,
             'overall_patch_loss':
             overall_patch_loss * opts.discriminator_loss_weight})


def discriminator_loss(rgb_img, true_dither):
    if len(rgb_img) != len(true_dither):
        raise Exception("expected equal number of RGB imgs & dithers")

    # discriminator loss is based on discriminator's ability to distinguish
    # (smoothed) true dithers ...
    smoothed_true_dither = (true_dither * 0.8) + 0.1
    discriminator_logits = discriminator(smoothed_true_dither, training=True)
    true_dither_loss = jnp.mean(discriminator_logits)

    # ...from fake dithers
    fake_dither = steep_sigmoid(generator(rgb_img))
    discriminator_logits = discriminator(fake_dither, training=True)
    fake_dither_loss = jnp.mean(discriminator_logits)

    # overall loss is the sum
    overall_loss = fake_dither_loss - true_dither_loss

    return (overall_loss,
            {'fake_dither_loss': fake_dither_loss,
             'true_dither_loss': true_dither_loss})


def build_train_step_fn(model, loss_fn):
    gradient_loss = objax.GradValues(loss_fn, model.vars())
    optimiser = objax.optimizer.Adam(model.vars())

    def train_step(learning_rate, rgb_img, true_dither):
        grads, _loss = gradient_loss(rgb_img, true_dither)
        grads = u.clip_gradients(grads, theta=opts.gradient_clip)
        optimiser(learning_rate, grads)
        grad_norms = [jnp.linalg.norm(g) for g in grads]
        return grad_norms

    if JIT:
        train_step = objax.Jit(
            train_step, gradient_loss.vars() + optimiser.vars())
    return train_step


generator_train_step = build_train_step_fn(
    generator, generator_loss)

discriminator_train_step = build_train_step_fn(
    discriminator, discriminator_loss)

# create learning rate utilities
generator_learning_rate = u.ValueFromFile(
    "generator_learning_rate.txt", 1e-3)
discriminator_learning_rate = u.ValueFromFile(
    "discriminator_learning_rate.txt", 1e-3)

# load some full res images for checking model performance during training
full_rgbs = []
full_true_dithers = []
for frame in [5000, 43000, 55000, 67000, 77000, 90000]:
    full_rgb, full_true_dither = data.parse_full_size(
        "frames/full_res/f_%08d.jpg" % frame)
    full_rgbs.append(full_rgb)
    full_true_dithers.append(full_true_dither)
full_rgbs = np.stack(full_rgbs)
full_true_dithers = np.stack(full_true_dithers)

# jit the generator now (we'll use it for predicting against the full res
# images) and also the two loss fns
if JIT:
    generator = objax.Jit(generator)
    generator_loss = objax.Jit(generator_loss, generator.vars())
    discriminator_loss = objax.Jit(discriminator_loss, discriminator.vars())

# setup output directory for full res samples
u.ensure_dir_exists("full_res_samples/%s" % RUN)
if os.path.exists("full_res_samples/latest"):
    os.remove("full_res_samples/latest")
os.symlink(RUN, "full_res_samples/latest")

# init dataset iterator
dataset = data.dataset(manifest_file=opts.manifest_file,
                       batch_size=opts.batch_size)

# set up ckpting for G and D
generator_ckpt = objax.io.Checkpoint(
    logdir=f"ckpts/{RUN}/generator/", keep_ckpts=20)
discriminator_ckpt = objax.io.Checkpoint(
    logdir=f"ckpts/{RUN}/discriminator/", keep_ckpts=20)

# run training loop!
for epoch in range(opts.epochs):

    generator_grads_min_max = None
    discriminator_grads_min_max = None

    # run some number of steps, alternating between training G and D
    train_generator = True
    for rgb_imgs, true_dithers in dataset.take(opts.steps_per_epoch):
        rgb_imgs = rgb_imgs.numpy()
        true_dithers = true_dithers.numpy()

        # collect grad norms once per epoch
        # if g_min_max is None:
        #     grad_norms = generator_train_step_with_grad_norms(
        #         learning_rate.value(), rgb_imgs, true_dithers)
        #     g_min_max = (float(jnp.min(grad_norms)),
        #                  float(jnp.max(grad_norms)))
        # else:
        if train_generator:
            grad_norms = generator_train_step(
                generator_learning_rate.value(), rgb_imgs, true_dithers)
            if generator_grads_min_max is None:
                generator_grads_min_max = (float(jnp.min(grad_norms)),
                                           float(jnp.max(grad_norms)))
        else:
            grad_norms = discriminator_train_step(
                discriminator_learning_rate.value(), rgb_imgs, true_dithers)
            if discriminator_grads_min_max is None:
                discriminator_grads_min_max = (float(jnp.min(grad_norms)),
                                               float(jnp.max(grad_norms)))
            # clip D weights. urgh; the is the hacky way to do the lipschitz
            # constraint; much better to get working with the gradient penalty
            # for k, v in discriminator.vars().items():
            #     print(k, jnp.min(v.value), jnp.max(v.value))
            for v in discriminator.vars().values():
                v.assign(jnp.clip(v.value,
                                  -opts.discriminator_weight_clip,
                                  opts.discriminator_weight_clip))
        train_generator = not train_generator

    # ckpt models
    generator_ckpt.save(generator.vars(), idx=epoch)
    discriminator_ckpt.save(discriminator.vars(), idx=epoch)

    # log range of values from generator
    fake_dither = steep_sigmoid(generator(rgb_imgs))
    range_of_dithers = jnp.around(jnp.percentile(
        fake_dither, jnp.linspace(0, 100, 11)), 2)

    # check loss against last batch
    overall_loss, component_losses = generator_loss(rgb_imgs, true_dithers)
    generator_losses = {
        'overall_loss': float(overall_loss),
        'reconstruction_loss': float(component_losses['reconstruction_loss']),
        'overall_patch_loss': float(component_losses['overall_patch_loss'])
    }
    overall_loss, component_losses = discriminator_loss(rgb_imgs, true_dithers)
    discriminator_losses = {
        'overall_loss': float(overall_loss),
        'fake_dither_loss': float(component_losses['fake_dither_loss']),
        'true_dither_loss': float(component_losses['true_dither_loss'])
    }

    # save full res pred dithers in a collage.
    full_pred_dithers = generator(full_rgbs)
    samples = [u.dither_to_pil(p) for p in full_pred_dithers]
    u.collage(samples).save("full_res_samples/%s/%05d.png" % (RUN, epoch))

    # some wandb logging
    wandb.log({
        'gen_overall_loss': generator_losses['overall_loss'],
        'gen_reconstruction_loss': generator_losses['reconstruction_loss'],
        'gen_overall_patch_loss': generator_losses['overall_patch_loss'],
        'discrim_overall_loss': discriminator_losses['overall_loss'],
        'discrim_fake_dither_loss': discriminator_losses['fake_dither_loss'],
        'discrim_true_dither_loss': discriminator_losses['true_dither_loss'],
        'generator_learning_rate': generator_learning_rate.value(),
        'discriminator_learning_rate': discriminator_learning_rate.value(),
        'generator_grad_norm_min': generator_grads_min_max[0],
        'generator_grad_norm_max': generator_grads_min_max[1],
        'discriminator_grad_norm_min': discriminator_grads_min_max[0],
        'discriminator_grad_norm_max': discriminator_grads_min_max[1]
    }, step=epoch)

    # some stdout logging
    print("epoch", epoch,
          "generator_learning_rate", generator_learning_rate.value(),
          "discriminator_learning_rate", discriminator_learning_rate.value(),
          "generator_losses", generator_losses,
          "generator_grads_min_max", generator_grads_min_max,
          "discriminator_losses", discriminator_losses,
          "discriminator_grads_min_max", discriminator_grads_min_max,
          "range_of_dithers", range_of_dithers)

    if finish_time is not None and time.time() > finish_time:
        break
