from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as tr
import tensorflow as tf
from tensorflow import keras
import typing

if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]


def train_step(model: tr.model.VanillaGAN, batch,
               g_opt: keras.optimizers.Optimizer,
               d_opt: keras.optimizers.Optimizer,
               record=False):
    feat = batch['feat']
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        gen, _, _ = model(feat, training=True, record=record)
        g_loss = model.losses[0]
        d_loss = model.losses[1]

        g_scope = model.generator.trainable_variables
        d_scope = model.discriminator.trainable_variables

        g_gradient = g_tape.gradient(g_loss, g_scope)
        d_gradient = d_tape.gradient(d_loss, d_scope)

        g_opt.apply_gradients(zip(g_gradient, g_scope))
        d_opt.apply_gradients(zip(d_gradient, d_scope))

    if record:
        tf.summary.image('train/image', img_helper(feat))
        tf.summary.image('train/gen', img_helper(gen))

    return g_loss.numpy(), d_loss.numpy()


def test_step(model: tr.model.VanillaGAN):
    x_gen = model.call_sample(batch_shape=[9, 100])
    tf.summary.image('test/gen', img_helper(x_gen))


def _map(x):
    img = x['image']
    x['feat'] = tf.cast(img, dtype=tf.float32) / 255.
    return x


def img_helper(x, row=3, column=3):
    img_num = row * column
    assert tf.shape(x)[0] >= img_num
    samples = x[:img_num, ...]
    samples = tf.reshape(samples, [row, column, 28, 28])
    samples = tf.transpose(samples, perm=[0, 2, 1, 3])
    return tf.reshape(samples, [row * 28, column * 28])[tf.newaxis, ..., tf.newaxis]


def main():
    model = tr.model.VanillaGAN(100)
    train_data, _ = tr.util.get_toy_data('mnist', 128, map_function=_map)
    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, .95)
    g_opt = keras.optimizers.Adam(lr)
    d_opt = keras.optimizers.Adam(lr)
    summary_path, save_path = tr.util.make_training_folder(ROOT_PATH, 'mnist_gan', 'vanilla_gan')
    writer = tf.summary.create_file_writer(summary_path)

    global_step = 0
    for epoch in range(100):
        for batch in train_data:
            with writer.as_default(step=global_step):
                record = global_step % 50 == 0
                g_loss, d_loss = train_step(model, batch, g_opt, d_opt, record=record)

                if global_step % 100 == 0:
                    print('epoch {}, step {}, g_loss {}, d_loss {}'.format(epoch, global_step, g_loss, d_loss))
                    test_step(model)

            global_step += 1


if __name__ == '__main__':
    main()
