from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

from talrasha.functional import relative_attention, scaled_dot_product_attention


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, v, k, q, mask, **kwargs):
        """

        :param v: value
        :param k: key
        :param q: query
        :param mask:
        :return:
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class RelativeAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, max_seq=2048):
        super(RelativeAttention, self).__init__(d_model, num_heads)
        self.max_seq = max_seq
        self.pos_emb = self.add_weight('pos_emb', shape=[self.max_seq, self.depth])

    def call(self, v, k, q, mask, **kwargs):
        """

        :param v: value
        :param k: key
        :param q: query
        :param mask:
        :return:
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # l_q = q.shape[2]
        l_q = tf.shape(q)[2]
        e = self._get_position_embedding(l_q)
        l_k = tf.shape(k)[2]
        q_et = tf.einsum('nhld,md->nhlm', q, e)
        q_et = self._masking(q_et)

        s = self._skew(q_et, l_q, l_k)

        scaled_attention, attention_weights = relative_attention(q, k, v, s, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def _get_position_embedding(self, len_q):
        print(len_q)
        # starting_point = tf.max(0, self.max_seq - len_q)
        starting_point = tf.math.maximum(0, self.max_seq - len_q)
        e = self.pos_emb[starting_point:, :]
        return e

    @staticmethod
    def _masking(qe: tf.Tensor):
        q_shape = tf.shape(qe)
        mask = tf.sequence_mask(
            tf.range(q_shape[-1] - 1, q_shape[-1] - q_shape[-2] - 1, -1), q_shape[-1])

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    @staticmethod
    def _skew(mat: tf.Tensor, l_q, l_k):
        """

        :param mat: [N H L L]
        :param l_q: length of query
        :param l_k: length of key
        :return:
        """
        mat_1 = tf.pad(mat, [[0, 0], [0, 0], [0, 0], [1, 0]], mode='CONSTANT')
        mat_shape = tf.shape(mat_1)
        head = mat_shape[-3]
        d_1 = mat_shape[-2]
        d_2 = mat_shape[-1]
        mat_1 = tf.reshape(mat_1, shape=[-1, head, d_2, d_1])
        s = mat_1[:, :, 1:, :]

        s = tf.cond(l_k > l_q, lambda: tf.pad(s, [[0, 0], [0, 0], [0, 0], [0, l_k - l_q]]), lambda: s)
        s = tf.cond(l_k < l_q, lambda: s[:, :, :, :l_k], lambda: s)

        # if l_k > l_q:
        #     s = tf.pad(s, [[0, 0], [0, 0], [0, 0], [0, l_k - l_q]])
        # elif l_k < l_q:
        #     s = s[:, :, :, :l_k]

        return s
