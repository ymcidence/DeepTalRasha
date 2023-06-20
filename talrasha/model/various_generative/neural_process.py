from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import typing

from talrasha.layer.basic_attention import MultiHeadAttention
from talrasha.functional import gaussian_kld, gaussian_prob
from talrasha.layer import GaussianReparameterization

from tensorflow import keras

if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


class _AttentionLayer(keras.Model):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = MultiHeadAttention(d_model, 1)

    # noinspection PyMethodOverriding
    def call(self, q, k, v, training=None, mask=None):
        return self.att(v, k, q, mask=mask, training=training)[0]


class AttentiveNP(keras.Model):

    def __init__(self, d_model=64, max_length=784, encoder=None, cross_attention=None, decoder=None, *args, **kwargs):
        """
        This is an implementation of [Attentive Neural Process](https://arxiv.org/pdf/1901.05761.pdf).
        The default settings point to an experiment that reconstructs MNIST images with y: the pixel value and x:
        the positions (with positional embeddings in the model).

        We only showcase dot-product attention in our implementation, but one can use some other ones by feeding an
        encoder when instantiating this model.

        The original paper uses 2 standalone encoders for the deterministic and stochastic paths respectively. To
        simplify the whole process, we use a shared attentional backbone with two different MLPs on the top
        for the two paths.

        By default, we use the normal distribution for both the likelihood, prior and posterior.

        :param d_model:
        :param max_length:
        :param encoder:
        :param cross_attention:
        :param decoder:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.max_length = max_length

        self.pos_emb = keras.layers.Embedding(max_length, d_model)
        self.encoder = encoder if encoder is not None else _AttentionLayer(d_model)
        self.decoder = decoder if decoder is not None else \
            keras.Sequential([
                keras.layers.Dense(2 * d_model, activation='relu'),
                keras.layers.Dense(2)
            ])

        self.cross_attention = cross_attention if cross_attention is not None else _AttentionLayer(d_model)
        self.fc_deterministic = keras.Sequential([
            keras.layers.LayerNormalization(),
            keras.layers.Dense(d_model)
        ])

        self.fc_stochastic = keras.Sequential([
            keras.layers.LayerNormalization(),
            keras.layers.Dense(d_model * 2)
        ])
        self.fc_value = keras.layers.Dense(d_model)
        self.rep = GaussianReparameterization(d_model)

    def _encoder_forward(self, pos, value, fc, training=True):
        x = pos + value
        feat, _ = self.encoder(x, training=training)

        return fc(feat, training=training)

    # noinspection PyMethodOverriding
    def call(self, context, target, training=None, mask=None, step=-1):
        c_pos = context[0]  # [N T1]
        c_pos_feat = self.pos_emb(c_pos)
        c_value = context[1][..., tf.newaxis]
        c_value_feat = self.fc_value(c_value, training=training)

        t_pos = target[0]
        t_pos_feat = self.pos_emb(t_pos)

        c_qkv = c_pos_feat + c_value_feat
        c_feat = self.encoder(c_qkv, c_qkv, c_qkv, mask=None, training=training)  # [N T1 D]
        c_feat_d = self.fc_deterministic(c_feat, training=training)  # [N T1 D]
        c_feat_s = tf.reduce_mean(self.fc_stochastic(c_feat, training=training), axis=1)  # [N 2D]
        c_mean, c_log_var = tf.split(c_feat_s, 2, axis=-1)  # [N D] * 2

        if target.__len__() > 1:
            t_value = target[1][..., tf.newaxis]
            t_value_feat = self.fc_value(t_value, training=training)

            t_qkv = t_pos_feat + t_value_feat

            t_feat = self.encoder(t_qkv, t_qkv, t_qkv, mask=None, training=training)
            t_feat_s = self.fc_stochastic(t_feat, training=training)
            t_feat_s = tf.reduce_mean(t_feat_s, axis=1)
            t_mean, t_log_var = tf.split(t_feat_s, 2, axis=-1)

            kld = tf.reduce_mean(
                tf.reduce_sum(gaussian_kld(t_mean, t_log_var, c_mean, c_log_var, reduce=False), axis=-1))
            if training:
                z = self.rep(t_mean, t_log_var)  # [N 1 D]
            else:
                z = self.rep(c_mean, c_log_var)  # [N 1 D]
        else:
            z = self.rep(c_mean, c_log_var)  # [N 1 D]
            kld = 0
            t_value = c_value

        r = self.cross_attention(t_pos_feat, c_pos_feat, c_feat_d, mask=None, training=training)  # [N T2 D] v, k, q

        decoder_input = t_pos_feat + r + z

        decoded = self.decoder(decoder_input, training=training)  # [N T2 2]

        y_mean, y_log_var = tf.split(decoded, 2, axis=-1)

        if target.__len__() > 1:

            likelihood = tf.reduce_mean(tf.reduce_sum(gaussian_prob(t_value, y_mean, y_log_var), axis=-1))
            elbo = likelihood - kld

            return y_mean, elbo
        else:
            return y_mean
