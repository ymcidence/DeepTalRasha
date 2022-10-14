from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from layer.group_transformer import GroupTransformer
from layer.transformer import MHA, positional_encoding, MLP

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


def dot_group_bias(q_group, k_group, group_emb):
    """

    :param q_group: [N H L1 G]
    :param k_group: [N H L2 G]
    :param group_emb: [G D]
    :return: [N H L1 L2]
    """
    q = tf.einsum('nhlg,gd->nhld', q_group, group_emb)
    k = tf.einsum('nhlg,gd->nhld', k_group, group_emb)

    bias = tf.einsum('nhld,nhsd->nhls', q, k)

    return bias


class BiasedMHA(MHA):
    # noinspection PyMethodOverriding
    def call(self, q, k, v, bias, training=None, mask=None):
        """

        :param q: [N L1 D]
        :param k: [N L2 D]
        :param v: [N L2 D]
        :param bias: [N H L1 L2] or [N 1 L1 L2] must be 4-dim
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
        soft_attention = tf.nn.softmax(soft_attention, axis=-1)

        biased_attention = (soft_attention + bias) / self.temp
        agg = tf.einsum('nhgl,nhld->nhgd', biased_attention, v)
        agg = tf.transpose(agg, [0, 2, 1, 3])  # [N G H d]
        agg = tf.reshape(agg, [batch_size, group_size, -1])  # [N G D] H*d=D
        agg = tf.nn.l2_normalize(agg, axis=-1)
        agg = self.fc_out(agg)

        return agg, soft_attention


class BiasedTransformer(keras.Model):
    def __init__(self, d_model, group_size, seq_length, head=1, drop_rate=.9, input_projection=True,
                 sinusoidal_pos=False, top_mlp=False, *args, **kwargs):
        """

        :param d_model: feature dim
        :param group_size: number of groups
        :param seq_length: number of patches
        :param head: number of heads (FOR MHA ONLY AND NOT FOR GROUPS)
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

        self.group_transform = GroupTransformer(d_model, group_size, seq_length, 1, drop_rate,
                                                input_projection=False, sinusoidal_pos=False, top_mlp=False)

        self.biased_mha = BiasedMHA(d_model, head)
        if input_projection:
            self.pos_emb = positional_encoding(seq_length, d_model) if sinusoidal_pos else tf.Variable(
                keras.initializers.TruncatedNormal()([seq_length, d_model]), trainable=True)
            self.fc_in = keras.layers.Dense(d_model)
        else:
            self.fc_in = None
            self.pos_emb = None

        self.modality_emb_1 = tf.Variable(keras.initializers.TruncatedNormal()([1, 1, d_model]), trainable=True)
        self.modality_emb_2 = tf.Variable(keras.initializers.TruncatedNormal()([1, 1, d_model]), trainable=True)

        self.drop = keras.layers.Dropout(drop_rate)
        if top_mlp:
            self.top_mlp = MLP(d_model, drop_rate, d_model)
        else:
            self.top_mlp = None

    # noinspection PyMethodOverriding
    def call(self, x_1, x_2, training=None, mask=None):
        """

        :param x_1: as query
        :param x_2: as key and value
        :param training:
        :param mask:
        :return rslt: feature to the top layers
        :return g_1: group feature for x_1
        :return g_2: group feature for x_2
        """
        if self.input_projection:
            x_1 = self.fc_in(x_1, training=training)
            x_2 = self.fc_in(x_2, training=training)
            x_1 = x_1 + self.pos_emb
            x_2 = x_2 + self.pos_emb

        g_1, soft_1, hard_1 = self.group_transform(x_1, training=training)
        g_2, soft_2, hard_2 = self.group_transform(x_2, training=training)

        self_q = x_1 + self.modality_emb_1  # [N L D]
        cross_q = self_q
        self_kv = x_1 + self.modality_emb_2
        cross_kv = x_2 + self.modality_emb_2  # [N L D]

        self_bias = dot_group_bias(hard_1, hard_1, self.group_transform.group_emb)  # [N 1 L L]
        cross_bias = dot_group_bias(hard_1, hard_2, self.group_transform.group_emb)  # [N 1 L L]

        self_reference = self.biased_mha(self_q, self_kv, self_kv, self_bias, training=training)
        cross_reference = self.biased_mha(cross_q, cross_kv, cross_kv, cross_bias, training=training)

        out_feat = self_reference + cross_reference
        out_feat = self.drop(out_feat)

        if self.top_mlp is not None:
            rslt = tf.nn.l2_normalize(out_feat + self.top_mlp(out_feat, training=training), axis=-1)
        else:
            rslt = out_feat
        return rslt, g_1, g_2
