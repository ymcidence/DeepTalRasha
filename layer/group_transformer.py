from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


def _hard_attention(s):
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


def _positional_encoding(position, d_model):
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)

    #
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    #
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class _MLP(keras.Model):
    def __init__(self, d_model, drop_rate, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = keras.Sequential([
            keras.layers.Dense(d_model, activation=keras.activations.gelu),
            keras.layers.Dropout(drop_rate),
            keras.layers.Dense(out_dim)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs, training=training)


class HardSetAttention(keras.Model):
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

    def _split_head(self, x):
        """

        :param x: [N L D]
        :return:
        """
        batch_size = tf.shape(x)[0]
        rslt = tf.reshape(x, [batch_size, -1, self.head, self.d_head])
        return tf.transpose(rslt, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, reduced_query=False, training=None, mask=None):
        """

        :param q: [G D] or [N G D]
        :param k: [N L D]
        :param v: [N L D]
        :param reduced_query: True-> q~[G D], False-> q~[N G D]
        :param training:
        :param mask:
        :return:
        """

        batch_size = tf.shape(v)[0]
        group_size = tf.shape(q)[-2]

        k = self._split_head(self.fc_k(k, training=training))  # [N H L d]
        v = self._split_head(self.fc_v(v, training=training))  # [N H L d]

        if reduced_query:
            _q = self._split_head(self.fc_q(q[tf.newaxis, :, :], training=training))  # [1 H G d]
            _q = tf.squeeze(_q, axis=0)
            soft_attention = tf.einsum('hgd,nhld->nhgl', _q, k) / self.temp  # [N H G L]
        else:
            _q = self._split_head(self.fc_q(q, training=training))  # [N H G d]
            soft_attention = tf.einsum('nhgd,nhld->nhgl', _q, k) / self.temp  # [N H G L]

        soft_attention = tf.nn.softmax(soft_attention, axis=-1)

        hard_attention = _hard_attention(soft_attention)

        agg = tf.einsum('nhgl,nhld->nhgd', hard_attention, v)
        agg = tf.transpose(agg, [0, 2, 1, 3])  # [N G H d]
        agg = tf.reshape(agg, [batch_size, group_size, -1])  # [N G D] H*d=D
        agg = tf.nn.l2_normalize(agg, axis=-1)
        agg = self.fc_out(agg)

        return agg, soft_attention, hard_attention


class GroupTransformer(keras.Model):
    def __init__(self, d_model, group_size, seq_length, head=1, drop_rate=.9, input_projection=True,
                 sinusoidal_pos=False, top_mlp=True, *args, **kwargs):
        """

        :param d_model: feature dim
        :param group_size: number of groups
        :param seq_length: number of patches
        :param head: number of heads
        :param drop_rate: the KEEP RATE of dropout
        :param input_projection: if the input needs an additional projection
        :param sinusoidal_pos:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.head = head
        self.d_model = d_model
        self.group_size = group_size
        self.seq_length = seq_length
        self.input_projection = input_projection
        self.group_attention = HardSetAttention(d_model, head=head)
        self.group_emb = tf.Variable(keras.initializers.TruncatedNormal()([group_size, d_model]), trainable=True)
        self.pos_emb = _positional_encoding(seq_length, d_model) if sinusoidal_pos else tf.Variable(
            keras.initializers.TruncatedNormal()([seq_length, d_model]), trainable=True)

        self.fc_in = keras.layers.Dense(d_model) if input_projection else None
        self.drop = keras.layers.Dropout(drop_rate)
        if top_mlp:
            self.top_mlp = _MLP(d_model, drop_rate, d_model)
        else:
            self.top_mlp = None

    def call(self, x, training=None, mask=None):
        if self.input_projection:
            x = self.fc_in(x, training=training)
        x += self.pos_emb

        group_feat, soft_attention, hard_attention = self.group_attention(self.group_emb, x, x, reduced_query=True,
                                                                          training=training)

        group_feat = tf.nn.l2_normalize(self.group_emb + group_feat)

        group_feat = self.drop(group_feat)

        if self.top_mlp is not None:
            rslt = tf.nn.l2_normalize(group_feat + self.top_mlp(group_feat, training=training), axis=-1)
        else:
            rslt = group_feat
        return rslt, soft_attention, hard_attention

