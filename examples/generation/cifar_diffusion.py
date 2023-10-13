from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as tr
import tensorflow as tf
from tensorflow import keras
import typing
from talrasha.model.diffusion._helpers import _UNet

# from typing import List
# import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]


# @tf.function
def train_step(model: keras.Model, batch, opt: keras.optimizers.Optimizer, step):
    feat = batch['feat']
    with tf.GradientTape() as tape:
        vlb = model(feat, training=True, step=step)

        gradients = tape.gradient(vlb, model.trainable_variables)
        clipped, g_norm = tf.clip_by_global_norm(gradients, 1.)
        opt.apply_gradients(zip(clipped, model.trainable_variables))

    if step >= 0:
        tf.summary.scalar('train/g_norm', g_norm, step=step)
    return vlb.numpy()


# @tf.function
def test_step(model: keras.Model, rng: tf.random.Generator, step):
    z = rng.normal(shape=[2, 32, 32, 3])
    img = model(z, training=False)
    img = [tf.reshape(v, [2, 32, 32, 3]) for v in img]
    # tf.summary.histogram('test/gen_hist', img[2], step=step)
    img = (tf.clip_by_value(tf.concat(img, axis=2), -1, 1) + 1.) / 2.
    tf.summary.image('test/img', img, max_outputs=2, step=step)


# @tf.function
def _map(x):
    # img = tf.image.resize(x['image'], [32, 32])
    # img =
    x['feat'] = 2 * tf.cast(x['image'], dtype=tf.float32) / 255. - 1
    return x


def main():
    backbone = _UNet(channel=3)
    model = tr.model.BasicDiffusion(1000, backbone=backbone)
    ds = tr.util.get_toy_data('cifar10', 128, map_function=_map)[0]
    opt = keras.optimizers.Adam(5e-4)
    summary_path, save_path = tr.util.make_training_folder(ROOT_PATH, 'cifar_gen', 'diffusion')
    writer = tf.summary.create_file_writer(summary_path)
    rng = tf.random.Generator.from_seed(22)

    with writer.as_default():
        step = 0
        for epoch in range(500):
            for batch in ds:
                summary_step = step if step % 200 == 0 else -1
                loss = train_step(model, batch, opt, summary_step)
                if summary_step >= 0:
                    print('epoch {}, step {}, loss {}'.format(epoch, step, loss))

                if step % 1000 == 0:
                    print('testing...')
                    test_step(model, rng, summary_step)
                    print('test finished')
                step += 1


if __name__ == '__main__':
    main()
