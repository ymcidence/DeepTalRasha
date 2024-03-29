from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

from talrasha.functional import sinusoidal_encoding
from .basic_attention import MultiHeadAttention, RelativeAttention


def _point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = _point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    # noinspection PyMethodOverriding
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class RelativeEncoderLayer(EncoderLayer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.mha = RelativeAttention(d_model, num_heads, max_seq=max_seq)
        self.ffn = _point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = _point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    # noinspection PyMethodOverriding
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class RelativeDecoderLayer(DecoderLayer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.mha1 = RelativeAttention(d_model, num_heads, max_seq=max_seq)
        self.mha2 = RelativeAttention(d_model, num_heads, max_seq=max_seq)

        self.ffn = _point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)


def layer_function(encode, d_model, num_heads, dff, rate, att='mha', **kwargs) -> keras.layers.Layer:
    if att == 'mha':
        return EncoderLayer(d_model, num_heads, dff, rate) if encode \
            else DecoderLayer(d_model, num_heads, dff, rate)
    elif att == 'rga':
        max_seq = kwargs.get('max_seq', 2048)
        return RelativeEncoderLayer(d_model, num_heads, dff, rate, max_seq) if encode \
            else RelativeDecoderLayer(d_model, num_heads, dff, rate, max_seq)
    else:
        raise NotImplementedError()


class BasicTransformerEncoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1,
                 att='mha', **kwargs):
        """

        :param num_layers: encoder layers
        :param d_model: feature dimension
        :param num_heads: number of the splits of multi-head attention
        :param dff: middle dimension for the feed forward network (2 fc layers)
        :param input_vocab_size:
        :param maximum_position_encoding: max sequence length
        :param rate: dropout rate (keep_prob = 1 - rate)
        :param att: type of attention
        """

        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = sinusoidal_encoding(maximum_position_encoding, self.d_model)[tf.newaxis, ...]

        self.enc_layers = [
            layer_function(True, d_model, num_heads, dff, rate, att=att, max_seq=maximum_position_encoding)
            for _ in range(num_layers)]

        self.dropout = keras.layers.Dropout(rate)

    # noinspection PyMethodOverriding
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        #
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class BasicTransformerDecoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, att='mha', **kwargs):
        """

        :param num_layers: decoder layers
        :param d_model: feature dimension
        :param num_heads: number of the splits of multi-head attention
        :param dff: middle dimension for the feed forward network (2 fc layers)
        :param target_vocab_size:
        :param maximum_position_encoding:
        :param rate: dropout rate (keep_prob = 1 - rate)
        :param att: type of attention
        """
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = sinusoidal_encoding(maximum_position_encoding, d_model)[tf.newaxis, ...]

        self.dec_layers = [
            layer_function(False, d_model, num_heads, dff, rate, att=att, max_seq=maximum_position_encoding)
            for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    # noinspection PyMethodOverriding
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
