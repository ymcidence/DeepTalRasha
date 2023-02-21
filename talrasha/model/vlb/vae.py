from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Optional

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

from talrasha.functional import *
from talrasha.layer import GaussianReparameterization


class _MNISTEncoder(keras.Model):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.network = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(d_model * 2),
        ])

    def call(self, x, training=True, mask=None):
        out = self.network(x, training=training)
        mean, log_var = tf.split(out, num_or_size_splits=2, axis=1)
        return mean, log_var


class _MNISTDecoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = keras.Sequential([
            tf.keras.layers.Dense(units=7 * 7 * 64, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
        ])

    def call(self, x, training=True, mask=None):
        batch_shape = tf.shape(x)
        if batch_shape.__len__() > 2:
            batch_size = batch_shape[0]
            sample_size = batch_shape[1]
            _x = tf.reshape(x, [batch_size * sample_size, -1])
            net_out = self.network(_x, training=training)

            out_shape = tf.shape(net_out)

            rslt = tf.reshape(net_out, [batch_size, sample_size, out_shape[-3], out_shape[-2], out_shape[-1]])
        else:
            rslt = self.network(x, training=training)
        return tf.nn.sigmoid(rslt)


class VanillaVAE(keras.Model):
    def __init__(self,
                 d_model: int,
                 sample_size=1,
                 encoder: Optional[keras.Model] = None,
                 decoder: Optional[keras.Model] = None,
                 likelihood='Bernoulli',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder if encoder is not None else _MNISTEncoder(d_model)
        self.decoder = decoder if decoder is not None else _MNISTDecoder()
        self.d_model = d_model
        self.sample_size = sample_size

        self.rep = GaussianReparameterization(d_model)
        self.likelihood = likelihood

    def call(self, x, training=True, mask=None, record=False):
        u_pzx, log_v_pzx = self.encoder(x, training=training)
        z_sample = self.rep(u_pzx, log_v_pzx, sample_size=self.sample_size, stochastic=True)  # [N L ... D]
        likelihood_parameter = self.decoder(z_sample, training=training)

        if self.likelihood == 'Gaussian':
            _mean, _log_var = likelihood_parameter
            log_pxz = gaussian_prob(x, _mean, _log_var, reduce=True)
        elif self.likelihood == 'Bernoulli':
            _mean = likelihood_parameter
            log_pxz = bernoulli_prob(x, likelihood_parameter, reduce=True)
        else:
            raise NotImplementedError('No likelihood function supported')

        kld = gaussian_kld(u_pzx, log_v_pzx, reduce=True)

        elbo = log_pxz - kld

        if training:
            self.add_loss(-elbo)

        if record:
            record_string = 'train/' if training else 'test/'
            tf.summary.scalar(record_string + 'elbo', elbo)
            tf.summary.scalar(record_string + 'kld', kld)
            tf.summary.scalar(record_string + 'likelihood', log_pxz)

        return _mean, elbo

    def call_sample(self, batch_size):
        batch_shape = [batch_size, self.d_model]
        z = tf.random.normal(batch_shape)

        likelihood_parameter = self.decoder(z, training=False)

        return likelihood_parameter if self.likelihood == 'Bernoulli' else likelihood_parameter[0]
