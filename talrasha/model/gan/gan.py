from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Optional

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


class _MNISTGenerator(keras.Model):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.network = keras.Sequential([
            keras.layers.Dense(7 * 7 * 256, input_dim=d_model),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Reshape([7, 7, 256]),
            keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         activation='sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.network(inputs, training=training)


class _MNISTDiscriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = keras.Sequential([
            keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.3),
            keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.3),
            keras.layers.Flatten(),
            keras.layers.Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.network(inputs, training=training)


class VanillaGAN(keras.Model):
    def __init__(self,
                 d_model,
                 generator: Optional[keras.Model] = None,
                 discriminator: Optional[keras.Model] = None,
                 *args, **kwargs):
        """

        Args:
            d_model: the latent size of the random input
            generator: a keras.Model
            discriminator: a keras.Model object. Must output a single logit.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.generator = _MNISTGenerator(d_model) if generator is None else generator
        self.discriminator = _MNISTDiscriminator() if discriminator is None else discriminator
        self.rng = tf.random.Generator.from_seed(0)

    def call(self, inputs, training=True, mask=None, record=False):
        batch_size = tf.shape(inputs)[0]
        z = self.rng.normal([batch_size, self.d_model])

        gen = self.generator(z, training=training)

        real_logit = self.discriminator(inputs, training=training)
        fake_logit = self.discriminator(gen, training=training)

        g_loss = keras.losses.binary_crossentropy(tf.ones_like(fake_logit), fake_logit, from_logits=True)
        g_loss = tf.reduce_mean(g_loss)

        d_loss_1 = keras.losses.binary_crossentropy(tf.zeros_like(fake_logit), fake_logit, from_logits=True)
        d_loss_2 = keras.losses.binary_crossentropy(tf.ones_like(real_logit), real_logit, from_logits=True)

        d_loss = d_loss_1 + d_loss_2
        d_loss = tf.reduce_mean(d_loss)

        if training:
            self.add_loss(g_loss)
            self.add_loss(d_loss)

        if record:
            record_string = 'train/' if training else 'test/'

            tf.summary.scalar(record_string + 'g_loss', g_loss)
            tf.summary.scalar(record_string + 'd_loss', d_loss)

        return gen, g_loss, d_loss

    @property
    def generator_scope(self):
        return self.generator.trainable_variables

    @property
    def discriminator_scope(self):
        return self.discriminator.trainable_variables

    def call_sample(self, batch_shape):
        z = self.rng.normal(batch_shape, dtype=tf.float32)
        return self.generator(z, training=False)
