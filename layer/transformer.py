from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


def hard_ste_attention(s):
    """
    One-hot attention + STE
    :param s: [N H G L]
    :return: [N H G L]
    """
    depth = tf.shape(s)[-2]
    assignment = tf.argmax(s, axis=-1)  # [N H L]

    hard_attention = tf.one_hot(assignment, depth=depth)  # [N H L G]
    hard_attention = tf.transpose(hard_attention, [0, 1, 3, 2])  # [N H G L]

    rslt = tf.stop_gradient(hard_attention) + s - tf.stop_gradient(s)
    return rslt


def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)

    #
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    #
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MLP(keras.Model):
    def __init__(self, d_model, drop_rate, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = keras.Sequential([
            keras.layers.Dense(d_model, activation=keras.activations.gelu),
            keras.layers.Dropout(drop_rate),
            keras.layers.Dense(out_dim)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs, training=training)


class MHA(keras.Model):
    def __init__(self, d_model, head=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = head
        self.d_model = d_model
        self.fc_q = keras.layers.Dense(d_model)
        self.fc_k = keras.layers.Dense(d_model)
        self.fc_v = keras.layers.Dense(d_model)
        self.d_head = d_model // head
        self.fc_out = keras.layers.Dense(d_model)
        self.temp = float(self.d_head) ** 0.5

    # noinspection PyMethodOverriding
    def call(self, q, k, v, training=None, mask=None):
        """

        :param q: [N L1 D]
        :param k: [N L2 D]
        :param v: [N L2 D]
        :param training:
        :param mask: not used
        :return:
        """
        batch_size = tf.shape(v)[0]
        group_size = tf.shape(q)[-2]

        q = self._split_head(self.fc_k(q, training=training))  # [N H L d]
        k = self._split_head(self.fc_k(k, training=training))  # [N H L d]
        v = self._split_head(self.fc_v(v, training=training))  # [N H L d]

        soft_attention = tf.einsum('nhgd,nhld->nhgl', q, k)
        soft_attention = tf.nn.softmax(soft_attention / self.temp, axis=-1)

        agg = tf.einsum('nhgl,nhld->nhgd', soft_attention, v)
        agg = tf.transpose(agg, [0, 2, 1, 3])  # [N G H d]
        agg = tf.reshape(agg, [batch_size, group_size, -1])  # [N G D] H*d=D
        agg = tf.nn.l2_normalize(agg, axis=-1)
        agg = self.fc_out(agg)

        return agg, soft_attention

    def _split_head(self, x):
        """

        :param x: [N L D]
        :return:
        """
        batch_size = tf.shape(x)[0]
        rslt = tf.reshape(x, [batch_size, -1, self.head, self.d_head])
        return tf.transpose(rslt, perm=[0, 2, 1, 3])
