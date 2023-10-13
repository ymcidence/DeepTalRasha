from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Optional

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

# noinspection PyProtectedMember
from talrasha.functional import *


class _MNISTEncoder(keras.Model):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.network = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(d_model),
        ])

    def call(self, x, training=True, mask=None):
        return self.network(x, training=training)


class _MNISTDecoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = keras.Sequential([
            keras.layers.Dense(units=7 * 7 * 64, activation=tf.nn.relu),
            keras.layers.Reshape(target_shape=(7, 7, 64)),
            keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
        ])

    def call(self, x, training=True, mask=None):
        rslt = self.network(x, training=training)
        return rslt


class WAEWithMMD(keras.Model):
    """
    This is the implementation of [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558).

    We only show the case of an MMD regularization in the middle.

    """
    def __init__(self,
                 d_model: int,
                 encoder: Optional[keras.Model] = None,
                 decoder: Optional[keras.Model] = None,
                 rkhs='imq',
                 mmd_weight=10.,
                 *args, **kwargs):
        """

        :param d_model: the LAST dimension of the latent space
        :param encoder:
        :param decoder:
        :param rkhs: the kernel name
        :param mmd_weight: the weight of the MMD term in the final loss
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.encoder = encoder if encoder is not None else _MNISTEncoder(d_model)
        self.decoder = decoder if decoder is not None else _MNISTDecoder()
        self.d_model = d_model
        self.rkhs = rkhs
        self.mmd_weight = mmd_weight
        self.rng = tf.random.Generator.from_seed(0)

    def call(self, inputs, training=True, mask=None, record=False):
        z = self.encoder(inputs, training=training)

        z_0 = self.rng.normal(tf.shape(z))

        decoded = self.decoder(z, training=training)

        mmd_loss = mmd(z, z_0, rkhs=self.rkhs)

        # batch_size = tf.shape(decoded)[-1]

        # l2_loss = tf.reduce_sum(tf.square(inputs - decoded)) / tf.cast(batch_size, tf.float32)

        l2_loss = tf.reduce_mean(tf.square(inputs - decoded))
        loss = l2_loss + self.mmd_weight * mmd_loss

        if training:
            self.add_loss(loss)

        if record:
            record_string = 'train/' if training else 'test/'
            tf.summary.scalar(record_string + 'loss', loss)
            tf.summary.scalar(record_string + 'mmd', mmd_loss)
            tf.summary.scalar(record_string + 'l2', l2_loss)

        return tf.clip_by_value(decoded, 0, 1)

    def call_sample(self, batch_shape):
        z = self.rng.normal(batch_shape)
        return tf.clip_by_value(self.decoder(z, training=False), 0, 1)
