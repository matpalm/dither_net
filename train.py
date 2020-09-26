import objax
import models
import jax.numpy as jnp
from objax.functional.loss import sigmoid_cross_entropy_logits
from PIL import Image
import numpy as np
import util as u
import wandb
import data

RUN = u.DTS()
FRAMES = [5000, 6500, 72000, 137000]

rgb_img = []
true_dither = []
for F in FRAMES:
    rgb, dither = data.parse("frames/f_%08d.jpg" % F)
    rgb_img.append(rgb)
    true_dither.append(dither)
rgb_img = jnp.stack(rgb_img)
true_dither = jnp.stack(true_dither)

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


def clip_gradients(grads, theta):
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(g) for g in grads])
    scale_factor = jnp.minimum(theta / total_grad_norm, 1.)
    return [g * scale_factor for g in grads]


def train_step(learning_rate, rgb_img, true_dither):
    grads, _loss = gradient_loss(rgb_img, true_dither)
    grads = clip_gradients(grads, 1)
    grad_norms = [jnp.linalg.norm(g) for g in grads]
    optimiser(learning_rate, grads)
    return grad_norms


train_step = objax.Jit(train_step,
                       gradient_loss.vars() + optimiser.vars())

learning_rate = 1e-3
# improvement_tracking = u.ImprovementTracking(patience=10, smoothing=0.5)

# for debug images
u.ensure_dir_exists("test/%s" % RUN)

for i in range(300):

    g_min = 1e6
    g_max = 0
    for _ in range(100):
        # for rgb_imgs, true_dithers in data.dataset(fname_glob='imgs/c2/*png',
        #                                            batch_size=4):
        grad_norms = train_step(learning_rate, rgb_img, true_dither)
        g_min = min(g_min, float(jnp.min(grad_norms)))
        g_max = max(g_max, float(jnp.max(grad_norms)))

    # check loss against sample images
    loss = float(cross_entropy(rgb_img, true_dither))

    # save dithers to disk
    sample_dithered_img = unet.dithers_as_pil(rgb_img)
    u.collage(sample_dithered_img).save("test/%s/%05d.png" % (RUN, i))

    # and prep images for wandb logging
    wand_imgs = []
    for d, f in zip(sample_dithered_img, FRAMES):
        wand_imgs.append(wandb.Image(d, caption="f_%05d" % f))
    wandb.log({'loss': loss, 'g_min': g_min, 'g_max': g_max,
               'learning_rate': learning_rate,
               'eg_img': wand_imgs}, step=i)

    # if not improvement_tracking.improved(loss):
    #     learning_rate /= 2
    #     improvement_tracking.reset()

    print("i", i, "lr", learning_rate, "g_min max", g_min, g_max, "loss", loss)
