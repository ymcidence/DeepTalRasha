from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as trs
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
    feat = batch['feat'] * 2 - 1.  # [-1, 1] normalization
    with tf.GradientTape() as tape:
        elbo = model(feat, training=True, step=step)

        gradients = tape.gradient(elbo, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    return elbo.numpy()


def test_step(model: keras.Model, step):
    z = tf.random.normal(shape=[1, 28 * 28])
    img = model(z, training=False)
    img = [tf.reshape(v, [1, 28, 28, 1]) for v in img]
    img = (tf.clip_by_value(tf.concat(img, axis=2), -1, 1) + 1.) / 2.
    tf.summary.image('test/img', img, step=step)


def main():
    model = trs.model.BasicDiffusion(100)
    mnist = trs.util.get_toy_data('mnist', 32)[0]
    opt = keras.optimizers.Adam(2e-4)
    summary_path, save_path = trs.util.make_training_folder(ROOT_PATH, 'mnist_diffusion')
    writer = tf.summary.create_file_writer(summary_path)

    with writer.as_default():
        step = 0
        for epoch in range(30):
            for batch in mnist:
                summary_step = step if step % 100 == 0 else -1
                loss = train_step(model, batch, opt, summary_step)
                if summary_step >= 0:
                    print('step {}, loss {}, start testing...'.format(step, loss))
                    test_step(model, summary_step)
                step += 1


if __name__ == '__main__':
    main()
