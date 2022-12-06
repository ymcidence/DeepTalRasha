from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as tr
import tensorflow as tf
from tensorflow import keras
import typing
from typing import List
import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]


def train_step(model: keras.Model, batch, opt: keras.optimizers.Optimizer, step):
    feat = batch['feat']
    with tf.GradientTape() as tape:
        elbo = model(feat, training=True, step=step)

        gradients = tape.gradient(elbo, model.trainable_variables)
        clipped, g_norm = tf.clip_by_global_norm(gradients, 1.)
        opt.apply_gradients(zip(clipped, model.trainable_variables))

    if step >= 0:
        tf.summary.scalar('train/g_norm', g_norm, step=step)
    return elbo.numpy()


def test_step(model: keras.Model, step):
    z = tf.random.normal(shape=[1, 28 * 28])
    img = model(z, training=False)
    img = [tf.reshape(v, [1, 28, 28, 1]) for v in img]
    tf.summary.histogram('test/gen_hist', img[2], step=step)
    img = (tf.clip_by_value(tf.concat(img, axis=2), -1, 1) + 1.) / 2.
    tf.summary.image('test/img', img, step=step)


def _map(x):
    img = x['image']
    x['feat'] = 2 * tf.cast(tf.reshape(img, [-1]), dtype=tf.float32) / 255. -1
    return x


def main():
    model = tr.model.BasicDiffusion(200)
    mnist = tr.util.get_toy_data('mnist', 64, map_function=_map)[0]
    opt = keras.optimizers.Adam(5e-4)
    summary_path, save_path = tr.util.make_training_folder(ROOT_PATH, 'mnist_gen', 'diffusion')
    writer = tf.summary.create_file_writer(summary_path)

    with writer.as_default():
        step = 0
        for epoch in range(100):
            for batch in mnist:
                summary_step = step if step % 200 == 0 else -1
                loss = train_step(model, batch, opt, summary_step)
                if summary_step >= 0:
                    print('epoch {}, step {}, loss {}'.format(epoch, step, loss))

                if step % 1000 == 0:
                    print('testing...')
                    test_step(model, summary_step)
                    print('test finished')
                step += 1


if __name__ == '__main__':
    main()
