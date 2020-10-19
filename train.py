import objax
import generator as g
import discriminator as d
import jax.numpy as jnp
from objax.functional import sigmoid
from PIL import Image
import numpy as np
import util as u
import wandb
import data
import argparse
import os
import time
import sys

JIT = True

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-file', type=str)
parser.add_argument('--group', type=str, default='dft')
parser.add_argument('--batch-size', type=int)
parser.add_argument('--gradient-clip', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--max-run-time', type=int, default=None,
                    help='max run time in secs')
parser.add_argument('--steps-per-epoch', type=int)
parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--positive-weight', type=float, default=1.0)
parser.add_argument('--reconstruction-loss-weight', type=float, default=1.0)
parser.add_argument('--change-loss-weight', type=float, default=0.0)
parser.add_argument('--discriminator-loss-weight', type=float, default=1.0)
parser.add_argument('--generator-sigmoid-b', type=float, default=1.0)
parser.add_argument('--discriminator-weight-clip', type=float, default=0.1)
parser.add_argument('--generator-learning-rate', type=float, default=1e-3)
parser.add_argument('--discriminator-learning-rate', type=float, default=1e-4)

opts = parser.parse_args()
print(opts)

finish_time = None
if opts.max_run_time is not None and opts.max_run_time > 0:
    finish_time = time.time() + opts.max_run_time

RUN = u.DTS()
print(">RUN", RUN)
sys.stdout.flush()

wandb.init(project='dither_net', group=opts.group, name=RUN)
wandb.config.gradient_clip = opts.gradient_clip
wandb.config.positive_weight = opts.positive_weight
wandb.config.reconstruction_loss_weight = opts.reconstruction_loss_weight
wandb.config.change_loss_weight = opts.change_loss_weight
wandb.config.discriminator_loss_weight = opts.discriminator_loss_weight
wandb.config.generator_sigmoid_b = opts.generator_sigmoid_b
wandb.config.discriminator_weight_clip = opts.discriminator_weight_clip
wandb.config.generator_learning_rate = opts.generator_learning_rate
wandb.config.discriminator_learning_rate = opts.discriminator_learning_rate

generator = g.Generator()
discriminator = d.Discriminator()

print("generator", generator.vars())
print("discriminator", discriminator.vars())
sys.stdout.flush()


def steep_sigmoid(x):
    # since D from true_dithers only sees (0, 1) we want to make it's
    # job harder for fake_dithers by squashing the sigmoid activation towards
    # 0 & 1. simplest way to do this is by varying B in the generalised
    # logisitic function. this is handy since that makes it just a function on
    # x, so we can continue to use the numerically stable jax.special.expit
    # that objax.sigmoid wraps.
    # see https://en.wikipedia.org/wiki/Generalised_logistic_function
    return sigmoid(opts.generator_sigmoid_b * x)


def generator_loss(rgb_img_t1, true_dither_t0, true_dither_t1):
    # generator loss is based on the generated images from the RGB and is
    # composed of two components...
    pred_dither_t1 = steep_sigmoid(generator(rgb_img_t1))

    # 1) a comparison to the t1 true_dither to see how well it reconstructs it
    per_pixel_reconstruction_loss = jnp.abs(pred_dither_t1 - true_dither_t1)
    loss_weight = jnp.where(true_dither_t1 == 1, opts.positive_weight, 1.0)
    reconstruction_loss = jnp.mean(loss_weight * per_pixel_reconstruction_loss)

    # 2) a comparison to the t0 true_dither to see how much has changed
    per_pixel_change_loss = jnp.abs(pred_dither_t1 - true_dither_t0)
    loss_weight = jnp.where(true_dither_t0 == 1, opts.positive_weight, 1.0)
    change_loss = jnp.mean(loss_weight * per_pixel_change_loss)

    # 3) how well it fools the discriminator
    discriminator_logits = discriminator(pred_dither_t1, training=False)
    overall_patch_loss = -jnp.mean(discriminator_logits)

    # overall loss is weighted combination of the two
    overall_loss = (reconstruction_loss * opts.reconstruction_loss_weight +
                    change_loss * opts.change_loss_weight +
                    overall_patch_loss * opts.discriminator_loss_weight)

    return (overall_loss,
            {'scaled': {'reconstruction_loss':
                        reconstruction_loss * opts.reconstruction_loss_weight,
                        'change_loss':
                        change_loss * opts.change_loss_weight,
                        'overall_patch_loss':
                        overall_patch_loss * opts.discriminator_loss_weight},
             'unscaled': {'reconstruction_loss': reconstruction_loss,
                          'change_loss': change_loss,
                          'overall_patch_loss': overall_patch_loss}})


def discriminator_loss(rgb_img, _true_dither_t0, true_dither):
    # TODO: https://trello.com/c/rKNz7oUv/41-d-doesnt-need-dithert0

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

    def train_step(learning_rate, rgb_img_t1, true_dither_t0, true_dither_t1):
        grads, _loss = gradient_loss(
            rgb_img_t1, true_dither_t0, true_dither_t1)
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
# generator_learning_rate = u.ValueFromFile(
#     "generator_learning_rate.txt", 1e-3)
# discriminator_learning_rate = u.ValueFromFile(
#     "discriminator_learning_rate.txt", 1e-3)

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
                       batch_size=opts.batch_size,
                       patch_size=opts.patch_size)

# set up ckpting for G and D
generator_ckpt = objax.io.Checkpoint(
    logdir=f"ckpts/{RUN}/generator/", keep_ckpts=20)
discriminator_ckpt = objax.io.Checkpoint(
    logdir=f"ckpts/{RUN}/discriminator/", keep_ckpts=20)

# TODO: D doesn't need _t1 images, so it's training step should pull from
# another generator only unpacking one pair of images

# run training loop!
for epoch in range(opts.epochs):

    generator_grads_min_max = None
    discriminator_grads_min_max = None

    # run some number of steps, alternating between training G and D
    train_generator = True
    for (rgb_imgs_t1, true_dithers_t0,
         true_dithers_t1) in dataset.take(opts.steps_per_epoch):

        rgb_imgs_t1 = rgb_imgs_t1.numpy()
        true_dithers_t0 = true_dithers_t0.numpy()
        true_dithers_t1 = true_dithers_t1.numpy()

        if train_generator:
            grad_norms = generator_train_step(
                opts.generator_learning_rate, rgb_imgs_t1, true_dithers_t0,
                true_dithers_t1)
            if generator_grads_min_max is None:
                generator_grads_min_max = (float(jnp.min(grad_norms)),
                                           float(jnp.max(grad_norms)))
        else:
            # TODO: https://trello.com/c/rKNz7oUv/41-d-doesnt-need-dithert0
            grad_norms = discriminator_train_step(
                opts.generator_learning_rate, rgb_imgs_t1, true_dithers_t0,
                true_dithers_t1)
            if discriminator_grads_min_max is None:
                discriminator_grads_min_max = (float(jnp.min(grad_norms)),
                                               float(jnp.max(grad_norms)))
            # clip D weights. urgh; this is the hacky way to do the lipschitz
            # constraint; much better to get working with the gradient penalty
            for v in discriminator.vars().values():
                v.assign(jnp.clip(v.value,
                                  -opts.discriminator_weight_clip,
                                  opts.discriminator_weight_clip))
        train_generator = not train_generator

    # ckpt models
    generator_ckpt.save(generator.vars(), idx=epoch)
    discriminator_ckpt.save(discriminator.vars(), idx=epoch)

    # check loss against last batch
    overall_loss, component_losses = generator_loss(
        rgb_imgs_t1, true_dithers_t0, true_dithers_t1)
    generator_losses = {  # clumsy o_O; treemap the float cast?
        'overall_loss': float(overall_loss),
        'scaled': {
            'change_loss':
                float(component_losses['scaled']['change_loss']),
            'reconstruction_loss':
                float(component_losses['scaled']['reconstruction_loss']),
            'overall_patch_loss':
                float(component_losses['scaled']['overall_patch_loss'])
        },
        'unscaled': {
            'change_loss':
                float(component_losses['unscaled']['change_loss']),
            'reconstruction_loss':
                float(component_losses['unscaled']['reconstruction_loss']),
            'overall_patch_loss':
                float(component_losses['unscaled']['overall_patch_loss'])
        }
    }
    overall_loss, component_losses = discriminator_loss(
        rgb_imgs_t1, true_dithers_t0, true_dithers_t1)
    discriminator_losses = {
        'overall_loss': float(overall_loss),
        'fake_dither_loss': float(component_losses['fake_dither_loss']),
        'true_dither_loss': float(component_losses['true_dither_loss'])
    }

    # save full res pred dithers in a collage.
    full_pred_dithers = generator(full_rgbs)
    samples = [u.dither_to_pil(p) for p in full_pred_dithers]
    collage = u.collage(samples)
    collage.save("full_res_samples/%s/%05d.png" % (RUN, epoch))
    collage.save("full_res_samples/last/%s.png" % RUN)

    # sanity check for collapse of all white or all black
    num_sample_white_pixels = int(jnp.sum(full_pred_dithers > 0))
    num_sample_black_pixels = int(jnp.sum(full_pred_dithers < 0))

    # some wandb logging
    wandb.log({
        'gen_overall_loss': generator_losses['overall_loss'],
        'gen_scaled_reconstruction_loss':
            generator_losses['scaled']['reconstruction_loss'],
        'gen_scaled_change_loss':
            generator_losses['scaled']['change_loss'],
        'gen_scaled_overall_patch_loss':
            generator_losses['scaled']['overall_patch_loss'],
        'gen_unscaled_reconstruction_loss':
            generator_losses['unscaled']['reconstruction_loss'],
        'gen_unscaled_change_loss':
            generator_losses['unscaled']['change_loss'],
        'gen_unscaled_overall_patch_loss':
            generator_losses['unscaled']['overall_patch_loss'],
        'discrim_overall_loss': discriminator_losses['overall_loss'],
        'discrim_fake_dither_loss': discriminator_losses['fake_dither_loss'],
        'discrim_true_dither_loss': discriminator_losses['true_dither_loss'],
        'generator_grad_norm_min': generator_grads_min_max[0],
        'generator_grad_norm_max': generator_grads_min_max[1],
        'discriminator_grad_norm_min': discriminator_grads_min_max[0],
        'discriminator_grad_norm_max': discriminator_grads_min_max[1],
        'num_sample_white_pixels': num_sample_white_pixels,
        'num_sample_black_pixels': num_sample_black_pixels
    }, step=epoch)

    # some stdout logging
    print("epoch", epoch,
          "generator_losses", generator_losses,
          "generator_grads_min_max", generator_grads_min_max,
          "discriminator_losses", discriminator_losses,
          "discriminator_grads_min_max", discriminator_grads_min_max,
          # "range_of_dithers", range_of_dithers,
          'num_sample_white_pixels', num_sample_white_pixels,
          'num_sample_black_pixels', num_sample_black_pixels)

    sys.stdout.flush()

    if finish_time is not None and time.time() > finish_time:
        print("time up", epoch)
        break

    if epoch >= 2 and (num_sample_white_pixels < 100000 or
                       num_sample_black_pixels < 100000):
        print("model collapse?", epoch)
        break
