from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
import typing
from tensorflow import keras
from functools import partial

import talrasha.functional

if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

from talrasha.layer import SiLU, MultiHeadAttention


class _ConvBlock(keras.Model):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = keras.layers.Conv2D(d_model, kernel_size=3, padding='SAME')
        self.bn = keras.layers.BatchNormalization()
        self.activation = SiLU()
        self.dropout = keras.layers.Dropout(.1)

    def call(self, x, training=None, mask=None, beta=None, gamma=None):
        x = self.conv(x, training=training)
        x = self.activation(x)
        if beta is not None and gamma is not None:
            x = x * (gamma + 1) + beta

        return self.dropout(self.activation(x))


class _ResBlock(keras.Model):
    def __init__(self, d_in, d_model, timed=False, *args, **kwargs):
        """

        :param d_in: dim of input
        :param d_model: dim of output
        :param timed: if a time emb is fed
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.mlp = keras.Sequential([
            SiLU(),
            keras.layers.Dense(d_model * 2)
        ]) if timed else None

        self.conv1 = _ConvBlock(d_model)
        self.conv2 = _ConvBlock(d_model)
        self.conv3 = keras.layers.Dense(d_model) if d_in != d_model else lambda x, training: x

    def call(self, inputs, time_emb=None, training=None, mask=None):
        if self.mlp is not None:
            assert time_emb is not None
            t = self.mlp(time_emb, training=training)[:, tf.newaxis, tf.newaxis, :]

            beta, gamma = tf.split(t, 2, axis=-1)
        else:
            beta = gamma = None

        z = self.conv1(inputs, training=training, gamma=gamma, beta=beta)
        z = self.conv2(z, training=training)

        return z + self.conv3(z, training=training)


class _Attention(keras.Model):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.net = MultiHeadAttention(d_model, num_heads)

    def call(self, inputs, training=None, mask=None):
        batch_shape = tf.shape(inputs)
        x = tf.reshape(inputs, [batch_shape[0], -1, batch_shape[-1]])
        x, _ = self.net(x, x, x, mask=mask)

        return tf.reshape(x, [*batch_shape[:-1], self.d_model])


class _Res(keras.Model):
    def __init__(self, transform: keras.Model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def call(self, x, training=None, mask=None):
        return self.transform(x, training=training) + x


class _PosEmb(keras.Model):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.positional_emb = partial(talrasha.functional.sinusoidal_encoding, d_model=d_model)

        self.positional_mlp = keras.Sequential([
            keras.layers.Dense(d_model * 4, activation=keras.activations.gelu),
            keras.layers.Dense(d_model * 4)
        ])

    def call(self, x, training=None, mask=None):
        x = self.positional_emb(x)
        return self.positional_mlp(x, training=training)


class _UNet(keras.Model):
    def __init__(self,
                 d_model=64,
                 d_initial=None,
                 d_output=None,
                 d_mul=(1, 2, 4, 8),
                 channel=3,
                 trainable_variance=False,
                 *args, **kwargs):
        """

        :param d_model:
        :param d_initial:
        :param d_output:
        :param d_mul:
        :param channel:
        :param trainable_variance:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.conv0 = keras.layers.Conv2D(42, kernel_size=7, padding='SAME')

        d_initial = d_model // 3 * 2 if d_initial is None else d_initial

        d_mid = [d_initial, *map(lambda x: d_model * x, d_mul)]
        in_out = list(zip(d_mid[:-1], d_mid[1:]))

        self.positional_mlp = _PosEmb(d_model)

        self.down_scaling = []
        self.up_scaling = []

        for i, (_in, _out) in enumerate(in_out):
            _attention = keras.Sequential([keras.layers.LayerNormalization(),
                                           _Attention(_out, num_heads=4)
                                           ])
            this_down_scaling = [_ResBlock(_in, _out, timed=True),
                                 _ResBlock(_out, _out, timed=True),
                                 _Res(_attention),
                                 ]
            if i < (len(in_out) - 1):
                this_down_scaling.append(keras.layers.Conv2D(_out, kernel_size=4, strides=2, padding='SAME'))

            self.down_scaling.append(this_down_scaling)

        mid_dim = d_mid[-1]
        mid_attention = keras.Sequential([keras.layers.LayerNormalization(),
                                          _Attention(mid_dim, num_heads=4)
                                          ])
        self.mid = [_ResBlock(mid_dim, mid_dim, timed=True),
                    _Res(mid_attention),
                    _ResBlock(mid_dim, mid_dim, timed=True)
                    ]

        for i, (_in, _out) in enumerate(reversed(in_out[1:])):
            _attention = keras.Sequential([keras.layers.LayerNormalization(),
                                           _Attention(_in, num_heads=4)
                                           ])
            this_up_scaling = [_ResBlock(_out * 2, _in, timed=True),
                               _ResBlock(_in, _in, timed=True),
                               _Res(_attention)
                               ]
            self.up_scaling.append(this_up_scaling)

            if i < (len(in_out) - 1):
                this_up_scaling.append(keras.layers.Conv2DTranspose(_in, kernel_size=4, strides=2, padding='SAME'))

            self.d_output = d_output if d_output is not None else channel * (2 if trainable_variance else 1)
            self.conv_final = keras.Sequential([_ResBlock(d_model * 2, d_model),
                                                keras.layers.Conv2D(self.d_output, 1)])

    def _encoding_call(self, x, t, training):
        h = []
        for f in self.down_scaling:
            x0 = f[0](x, t, training=training)
            x1 = f[1](x0, t, training=training)
            x2 = f[2](x1, training=training)
            h.append(x2)
            if len(f) > 3:
                x = f[3](x2, training=training)
            else:
                x = x2

        return x, h

    def _decoding_call(self, x, t, h, training):
        for f in self.up_scaling:
            x = tf.concat([x, h.pop()], axis=-1)
            x0 = f[0](x, t, training=training)
            x1 = f[1](x0, t, training=training)
            x2 = f[2](x1, training=training)
            if len(f) > 3:
                x = f[3](x2, training=training)
            else:
                x = x2
        return x, h

    def _mid_call(self, x, t, training):
        x = self.mid[0](x, t, training=training)
        x = self.mid[1](x, training=training)
        return self.mid[2](x, t, training=training)

    # noinspection PyMethodOverriding
    def call(self, x, t, training=None, mask=None):
        x = self.conv0(x, training=training)
        t = self.positional_mlp(t, training=training)

        x, h = self._encoding_call(x, t, training=training)
        x = self._mid_call(x, t, training=training)
        x, h = self._decoding_call(x, t, h, training=training)

        x = tf.concat([x, h.pop()], axis=-1)
        return self.conv_final(x, training=training)
