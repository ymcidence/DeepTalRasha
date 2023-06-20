from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from talrasha.util.toy_data import get_toy_data

AUTOTUNE = tf.data.experimental.AUTOTUNE


def gen_indices(col_ind):
    """

    :param col_ind: [N D 1]
    :return:
    """
    shape = tf.shape(col_ind)
    row_ind = tf.range(0, shape[0])
    row_ind = tf.expand_dims(tf.tile(tf.expand_dims(row_ind, -1), [1, shape[1]]), -1)
    ind = tf.concat([row_ind, col_ind], axis=-1)
    return ind


def plot_mnist(o_x, o_y, t_y, gt_y, size=28):
    batch_size = tf.shape(o_x)[0]
    target_shape = [batch_size, size, size, 1]
    original_shape = [batch_size, size * size]

    # _o_x = tf.cast((o_x + 1) / 2 * (size * size), tf.int32)
    _o_x = tf.cast(o_x, tf.int32)
    indices = gen_indices(_o_x)
    _o_y = tf.scatter_nd(indices, tf.squeeze(o_y), original_shape)

    p_1 = tf.reshape(_o_y, target_shape)
    p_2 = tf.reshape(gt_y, target_shape)
    p_3 = tf.reshape(t_y, target_shape)

    return tf.concat([p_1, p_2, p_3], axis=2)


class MaskedMNIST(object):
    def __init__(self, batch_size=64, max_ob=256):
        """

        :param batch_size:
        :param max_ob:
        """
        # data = tf.keras.datasets.mnist.load_data()
        # (x_train, self.y_train), (x_test, self.y_test) = data
        # x_train, x_test = x_train / 255., x_test / 255.

        self.batch_size = batch_size
        self.dim = 28 * 28
        self.max_ob = max_ob
        self.rng = tf.random.Generator.from_seed(0)
        self._build()

    def _build(self):
        mnist = get_toy_data('mnist', self.batch_size)

        def parser(d):
            _x = tf.cast(d['feat'], tf.float32)
            _l = tf.cast(d['label'], tf.int32)

            num_ob = self.rng.uniform(shape=(), minval=64, maxval=self.max_ob, dtype=tf.int32)

            ob_pos = self.rng.uniform([self.batch_size, num_ob], 0, self.dim - 1, dtype=tf.int32)
            ob_pos = tf.expand_dims(ob_pos, -1)

            indices = gen_indices(ob_pos)
            ob_value = tf.gather_nd(_x, indices)
            # ob_pos = (tf.cast(ob_pos, tf.float32) * 2. / self.dim) - 1

            tar_pos = tf.range(0, self.dim, dtype=tf.int32)
            tar_pos = tf.tile(tf.expand_dims(tar_pos, axis=0), [self.batch_size, 1])
            tar_value = _x
            rslt = [tf.squeeze(ob_pos, axis=-1), ob_value, tar_pos, tar_value, _l]
            return rslt

        data_train = mnist[0]
        data_test = mnist[1]

        if self.max_ob >= 0:
            self.data_train = data_train.map(parser, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
            self.data_test = data_test.map(parser, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        else:
            self.data_train = data_train
            self.data_test = data_test
