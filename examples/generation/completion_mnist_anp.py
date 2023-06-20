from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as tr
import tensorflow as tf
from tensorflow import keras
import typing

from talrasha.util.masked_data import plot_mnist, MaskedMNIST

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]


def train_step(model: tr.model.AttentiveNP, data, opt: tf.optimizers.Optimizer, record, step=0):
    o_x, o_y, t_x, t_y, label = data
    with tf.GradientTape() as tape:
        mean, elbo = model([o_x, o_y], [t_x, t_y], training=True)
        loss = -elbo
        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    if record:
        tf.summary.scalar('train/elbo', elbo, step=step)
        img = plot_mnist(o_x[..., tf.newaxis], o_y[..., tf.newaxis], mean, t_y[..., tf.newaxis])
        tf.summary.image('test/plot', img, step, max_outputs=1)

    return loss.numpy()


def test_step(model: tr.model.AttentiveNP, data, step):
    o_x, o_y, t_x, t_y, label = data
    mean, elbo = model([o_x, o_y], [t_x, t_y], training=False)
    img = plot_mnist(o_x[..., tf.newaxis], o_y[..., tf.newaxis], mean, t_y[..., tf.newaxis])
    tf.summary.scalar('test/elbo', elbo, step=step)
    tf.summary.image('test/plot', img, step, max_outputs=1)
    return


def main():
    model = tr.model.AttentiveNP()
    data = MaskedMNIST()
    train_data = data.data_train
    test_data = data.data_test
    test_data = test_data.repeat()
    test_iter = iter(test_data)
    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, .9)
    opt = keras.optimizers.Adam(lr)
    summary_path, save_path = tr.util.make_training_folder(ROOT_PATH, 'mnist_gen', 'cat_vae_bin2')
    writer = tf.summary.create_file_writer(summary_path)

    global_step = 0

    for epoch in range(100):
        for batch in train_data:
            with writer.as_default(step=global_step):
                record = global_step % 50 == 0
                loss = train_step(model, batch, opt, record=record, step=global_step)

                if global_step % 100 == 0:
                    print('epoch {}, step {}, loss {}'.format(epoch, global_step, loss))
                    test_step(model, next(test_iter), step=global_step)

            global_step += 1


if __name__ == '__main__':
    main()
