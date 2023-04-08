from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as tr
import tensorflow as tf
from tensorflow import keras
import typing

# from typing import List
# import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]


def train_step(model: keras.Model, batch, opt: keras.optimizers.Optimizer, record=False):
    feat = batch['feat']
    with tf.GradientTape() as tape:
        x_hat, elbo = model(feat, training=True, record=record)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    if record:
        tf.summary.image('train/image', img_helper(feat))
        tf.summary.image('train/recon', img_helper(x_hat[:, 0, ...]))
    return loss.numpy()


def test_step(model: keras.Model, batch):
    feat = batch['feat']
    x_hat, _ = model(feat, training=False, record=True)
    x_gen = model.call_sample(9)
    tf.summary.image('test/recon', img_helper(x_hat[:, 0, ...]))
    tf.summary.image('test/gen', img_helper(x_gen))


def img_helper(x, row=3, column=3):
    img_num = row * column
    assert tf.shape(x)[0] >= img_num
    samples = x[:img_num, ...]
    samples = tf.reshape(samples, [row, column, 28, 28])
    samples = tf.transpose(samples, perm=[0, 2, 1, 3])
    return tf.reshape(samples, [row * 28, column * 28])[tf.newaxis, ..., tf.newaxis]


def _map(x):
    img = x['image']
    x['feat'] = tf.cast(img, dtype=tf.float32) / 255.
    return x


def main():
    model = tr.model.VanillaVAE(16, sample_size=1)
    train_data, test_data = tr.util.get_toy_data('mnist', 128, map_function=_map)
    test_data = train_data.repeat()
    test_iter = iter(test_data)
    opt = keras.optimizers.Adam(1e-4)
    summary_path, save_path = tr.util.make_training_folder(ROOT_PATH, 'mnist_gen', 'vae')
    writer = tf.summary.create_file_writer(summary_path)

    global_step = 0

    for epoch in range(20):
        for batch in train_data:
            with writer.as_default(step=global_step):
                record = global_step % 50 == 0
                loss = train_step(model, batch, opt, record=record)

                if global_step % 100 == 0:
                    print('epoch {}, step {}, loss {}'.format(epoch, global_step, loss))
                    test_step(model, next(test_iter))

            global_step += 1


if __name__ == '__main__':
    main()
