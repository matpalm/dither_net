import objax
import models
import jax.numpy as jnp
from objax.functional import sigmoid
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
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--steps-per-epoch', type=int)
parser.add_argument('--positive-weight', type=float, default=1.0)
parser.add_argument('--reconstruction-loss-weight', type=float, default=1.0)
parser.add_argument('--discriminator-loss-weight', type=float, default=1.0)
opts = parser.parse_args()
print(opts)

RUN = u.DTS()

wandb.init(project='dither_net', group='v1', name=RUN)

generator = models.Generator()
discriminator = models.Discriminator()

print("generator", generator.vars())
print("discriminator", discriminator.vars())


def generator_loss(rgb_img, true_dither):
    # generator loss is based on the generated images from the RGB and is
    # composed of two components...
    pred_dither_logits = generator(rgb_img)
#     print("pred_dither_logits", pred_dither_logits.shape)

    # 1) a comparison to the true_dither to see how well it reconstructs it
    per_pixel_reconstruction_loss = sigmoid_cross_entropy_logits(
        pred_dither_logits, true_dither)
#     print("per_pixel_reconstruction_loss.shape", per_pixel_reconstruction_loss.shape)
    loss_weight = jnp.where(true_dither == 1, opts.positive_weight, 1.0)
    reconstruction_loss = jnp.mean(loss_weight * per_pixel_reconstruction_loss)

    # 2) how well it fools the discriminator
    discriminator_logits = discriminator(sigmoid(pred_dither_logits),
                                         training=False)
#     print("discriminator_result.shape", discriminator_result.shape)
    per_patch_loss = sigmoid_cross_entropy_logits(
        logits=discriminator_logits,
        labels=jnp.ones_like(discriminator_logits))
#     print("per_patch_loss.shape", per_patch_loss.shape)
    overall_patch_loss = jnp.mean(per_patch_loss)

    # overall loss is weighted combination of the two
    overall_loss = (reconstruction_loss * opts.reconstruction_loss_weight +
                    overall_patch_loss * opts.discriminator_loss_weight)

    return (overall_loss,
            {'reconstruction_loss': reconstruction_loss,
             'overall_patch_loss': overall_patch_loss})


def clip_gradients(grads, theta):
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
    scale_factor = jnp.minimum(theta / total_grad_norm, 1.)
    return [g * scale_factor for g in grads]


generator_gradient_loss = objax.GradValues(generator_loss, generator.vars())

generator_optimiser = objax.optimizer.Adam(generator.vars())


# def train_generator_step(learning_rate, rgb_img, true_dither):
#     grads, _loss = gradient_loss(rgb_img, true_dither)
#     grads = clip_gradients(grads, theta=opts.gradient_clip)
#     optimiser(learning_rate, grads)


# def generator_train_step_with_grad_norms(learning_rate, rgb_img, true_dither):
#     grads, _loss = generator_gradient_loss(rgb_img, true_dither)
#     grads = clip_gradients(grads, theta=opts.gradient_clip)
#     generator_optimiser(learning_rate, grads)
#     grad_norms = [jnp.linalg.norm(g) for g in grads]
#     return grad_norms

def generator_train_step(learning_rate, rgb_img, true_dither):
    grads, _loss = generator_gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, theta=opts.gradient_clip)
    generator_optimiser(learning_rate, grads)


generator_train_step = objax.Jit(generator_train_step,
                                 generator_gradient_loss.vars() + generator_optimiser.vars())


def discriminator_loss(rgb_img, true_dither):
    if len(rgb_img) != len(true_dither):
        raise Exception("expected equal number of RGB imgs & dithers")

    # discriminator loss is based on discriminator's ability to distinguish
    # fake dithers...
    fake_dither = sigmoid(generator(rgb_img))
    discriminator_logits = discriminator(fake_dither, training=True)
    # print("FAKE discriminator_logits", jnp.around(
    #     sigmoid(discriminator_logits).flatten(), 2))
    fake_dither_loss = sigmoid_cross_entropy_logits(
        logits=discriminator_logits,
        labels=jnp.zeros_like(discriminator_logits))

    # ... from true dithers
    discriminator_logits = discriminator(true_dither, training=True)
    # print("TRUE discriminator_logits", jnp.around(
    #     sigmoid(discriminator_logits).flatten(), 2))
    true_dither_loss = sigmoid_cross_entropy_logits(
        logits=discriminator_logits,
        labels=jnp.ones_like(discriminator_logits))

    # overall loss is the mean across them all
    overall_loss = jnp.mean(jnp.concatenate([fake_dither_loss.flatten(),
                                             true_dither_loss.flatten()]))

    return (overall_loss,
            {'fake_dither_loss': jnp.mean(fake_dither_loss),
             'true_dither_loss': jnp.mean(true_dither_loss)})


discriminator_gradient_loss = objax.GradValues(
    discriminator_loss, discriminator.vars())

discriminator_optimiser = objax.optimizer.Adam(discriminator.vars())


def discriminator_train_step(learning_rate, rgb_img, true_dither):
    grads, _loss = discriminator_gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, theta=opts.gradient_clip)
    discriminator_optimiser(learning_rate, grads)


discriminator_train_step = objax.Jit(discriminator_train_step,
                                     discriminator_gradient_loss.vars() + discriminator_optimiser.vars())


def predict(rgb_imgs):
    return generator(rgb_imgs, training=False)


predict = objax.Jit(predict, generator.vars())


generator_learning_rate = u.ValueFromFile(
    "generator_learning_rate.txt", 1e-3)
discriminator_learning_rate = u.ValueFromFile(
    "discriminator_learning_rate.txt", 1e-3)

# improvement_tracking = u.ImprovementTracking(patience=10, smoothing=0.5)

# for debug images
u.ensure_dir_exists("test/%s" % RUN)
if os.path.exists("test/latest"):
    os.remove("test/latest")
os.symlink(RUN, "test/latest")

# load some full res images for checking model performance
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

generator_ckpt = objax.io.Checkpoint(
    logdir=f"ckpts/{RUN}/generator/", keep_ckpts=20)
discriminator_ckpt = objax.io.Checkpoint(
    logdir=f"ckpts/{RUN}/discriminator/", keep_ckpts=20)

for epoch in range(opts.epochs):

    # g_min_max = None

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
            generator_train_step(
                generator_learning_rate.value(), rgb_imgs, true_dithers)
        else:
            discriminator_train_step(
                discriminator_learning_rate.value(), rgb_imgs, true_dithers)

        train_generator = not train_generator

    # ckpt
    generator_ckpt.save(generator.vars(), idx=epoch)
    discriminator_ckpt.save(discriminator.vars(), idx=epoch)

    # check loss against last batch as well as sample batch
    # TODO: ensure jitted
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

    # save sample rgb, true dith and pred dither in collage.
    if epoch % 10 == 0:
        full_pred_dithers = predict(full_rgbs)
        samples = []
        for r, t, p in zip(full_rgbs, full_true_dithers, full_pred_dithers):
            triplet = [u.rgb_img_to_pil(r), u.dither_to_pil(t),
                       u.dither_to_pil(p)]
            samples.append(u.collage(triplet, side_by_side=True))
        u.collage(samples).save("test/%s/%05d.png" % (RUN, epoch))

    # some wandb logging
    wandb.log({
        'gen_overall_loss': generator_losses['overall_loss'],
        'gen_reconstruction_loss': generator_losses['reconstruction_loss'],
        'gen_overall_patch_loss': generator_losses['overall_patch_loss'],
        'discrim_overall_loss': discriminator_losses['overall_loss'],
        'discrim_fake_dither_loss': discriminator_losses['fake_dither_loss'],
        'discrim_true_dither_loss': discriminator_losses['true_dither_loss'],
        'generator_learning_rate': generator_learning_rate.value(),
        'discriminator_learning_rate': discriminator_learning_rate.value()
    }, step=epoch)

    # if not improvement_tracking.improved(loss):
    #     learning_rate /= 2
    #     improvement_tracking.reset()

    print("epoch", epoch,
          "generator_learning_rate", generator_learning_rate.value(),
          "discriminator_learning_rate", discriminator_learning_rate.value(),
          "generator_losses", generator_losses,
          'discriminator_losses', discriminator_losses)
