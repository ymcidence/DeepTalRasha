from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from talrasha.layer.transformer import MHA, hard_ste_attention, positional_encoding, MLP

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


class HardSetAttention(MHA):

    # noinspection PyMethodOverriding
    def call(self, q, k, v, reduced_query=False, training=None, mask=None):
        """

        :param q: [G D] or [N G D]
        :param k: [N L D]
        :param v: [N L D]
        :param reduced_query: True-> q~[G D], False-> q~[N G D]
        :param training:
        :param mask:
        :return: [N G D] [N H G L] [N H G L]
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

        hard_attention = hard_ste_attention(soft_attention)

        agg = tf.einsum('nhgl,nhld->nhgd', hard_attention, v)
        agg = tf.transpose(agg, [0, 2, 1, 3])  # [N G H d]
        agg = tf.reshape(agg, [batch_size, group_size, -1])  # [N G D] H*d=D
        agg = tf.nn.l2_normalize(agg, axis=-1)
        agg = self.fc_out(agg)

        return agg, soft_attention, hard_attention


class GroupTransformer(keras.Model):
    def __init__(self, d_model, group_size, seq_length, head=1, drop_rate=.9, input_projection=True,
                 sinusoidal_pos=False, mlp=False, top_size=3, *args, **kwargs):
        """

        :param d_model: feature dim
        :param group_size: number of groups
        :param seq_length: number of patches
        :param head: number of heads
        :param drop_rate: the KEEP RATE of dropout
        :param input_projection: if the input needs an additional projection
        :param sinusoidal_pos: only valid when input_projection=True
        :param mlp: if an MLP is needed on the top of attention
        :param top_size:
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

        if input_projection:
            self.pos_emb = positional_encoding(seq_length, d_model) if sinusoidal_pos else tf.Variable(
                keras.initializers.TruncatedNormal()([seq_length, d_model]), trainable=True)
            self.fc_in = keras.layers.Dense(d_model)
        else:
            self.fc_in = None
            self.pos_emb = None

        self.drop = keras.layers.Dropout(drop_rate)
        if mlp:
            self.mlp = MLP(d_model, drop_rate, top_size)
        else:
            self.mlp = None

        if top_size > 0:
            self.fc_top = keras.layers.Dense(top_size)
        else:
            self.fc_top = None

    def call(self, x, training=None, mask=None):
        if self.input_projection:
            x = self.fc_in(x, training=training)
            x += self.pos_emb

        group_feat, soft_attention, hard_attention = self.group_attention(self.group_emb, x, x, reduced_query=True,
                                                                          training=training)

        group_feat = tf.nn.l2_normalize(self.group_emb + group_feat)

        group_feat = self.drop(group_feat)

        if self.mlp is not None:
            rslt = tf.nn.l2_normalize(group_feat + self.mlp(group_feat, training=training), axis=-1)
        else:
            rslt = group_feat

        if self.fc_top is not None:
            rslt = self.fc_top(rslt, training=training)
        return rslt, soft_attention, hard_attention
