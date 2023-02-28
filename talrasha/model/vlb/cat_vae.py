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
from talrasha.layer import GumbelReparameterization


class _MNISTEncoder(keras.Model):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.network = keras.Sequential([
            keras.layers.Flatten(),
            # keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(20 * d_model),
            keras.layers.Reshape([20, d_model])
        ])

    def call(self, x, training=True, mask=None):
        out = self.network(x, training=training)
        return out


class _MNISTDecoder(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(784),
            keras.layers.Reshape([28, 28, 1]),
        ])

    def call(self, x, training=True, mask=None):
        batch_shape = tf.shape(x)
        if batch_shape.__len__() > 3:  # which means multiple samples are stacked along the 2nd axis, i.e., [N S HW C]
            batch_size = batch_shape[0]
            sample_size = batch_shape[1]
            _x = tf.reshape(x, [batch_size * sample_size, batch_shape[-2], batch_shape[-1]])
            net_out = self.network(_x, training=training)

            out_shape = tf.shape(net_out)

            rslt = tf.reshape(net_out, [batch_size, sample_size, out_shape[-3], out_shape[-2], out_shape[-1]])
        else:
            rslt = self.network(x, training=training)
        return tf.nn.sigmoid(rslt)


class _CustomSchedule(object):
    def __init__(self, initial_value, anneal_rate, min_value, decay_step=1000):
        self.initial_value = initial_value
        self.anneal_rate = tf.constant(anneal_rate, tf.float32)

        self.min_value = tf.constant(min_value, tf.float32)
        self.decay_step = decay_step

    def __call__(self, step):
        _i = tf.cast(tf.math.floor(step / self.decay_step) * self.decay_step, tf.float32)
        rslt = self.initial_value * tf.exp(-self.anneal_rate * _i)

        return tf.maximum(rslt, self.min_value)


class CategoricalVAE(keras.Model):
    def __init__(self,
                 d_model: int,
                 sample_size=1,
                 temp=1.,
                 encoder: Optional[keras.Model] = None,
                 decoder: Optional[keras.Model] = None,
                 likelihood='Bernoulli',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder if encoder is not None else _MNISTEncoder(d_model)
        self.decoder = decoder if decoder is not None else _MNISTDecoder()
        self.d_model = d_model
        self.sample_size = sample_size
        self.temp = temp
        self.temp_schedule = _CustomSchedule(temp, anneal_rate=3e-5, min_value=.5)
        self.rep = GumbelReparameterization(d_model, temp=temp)
        self.likelihood = likelihood

    def call(self, x, training=None, mask=None, record=False, step=0):
        _temp = self.temp_schedule(step)
        cat_logit = self.encoder(x, training=training)
        z_sample = self.rep.call(cat_logit, sample_size=self.sample_size, stochastic=training, temp=_temp)
        likelihood_parameter = self.decoder(z_sample, training=training)

        if self.likelihood == 'Gaussian':
            _mean, _log_var = likelihood_parameter
            log_pxz = gaussian_prob(x, _mean, _log_var, reduce=True)
        elif self.likelihood == 'Bernoulli':
            _mean = likelihood_parameter
            log_pxz = bernoulli_prob(x, likelihood_parameter, reduce=True)
        else:
            raise NotImplementedError('No likelihood function supported')

        kld = categorical_kld(tf.nn.softmax(cat_logit), reduce=True)

        elbo = log_pxz - kld

        if training:
            self.add_loss(-elbo)

        if record:
            logic_choice = tf.cast(tf.reshape(tf.argmax(cat_logit, axis=-1), [-1, 1]), tf.int32).numpy()
            sample_choice = tf.cast(tf.reshape(tf.argmax(z_sample, axis=-1), [-1, 1]), tf.int32).numpy()
            record_string = 'train/' if training else 'test/'
            tf.summary.histogram(record_string + 'logic_choice', logic_choice, buckets=self.d_model)
            tf.summary.histogram(record_string + 'sample_choice', sample_choice, buckets=self.d_model)
            tf.summary.scalar(record_string + 'elbo', elbo)
            tf.summary.scalar(record_string + 'kld', kld)
            tf.summary.scalar(record_string + 'likelihood', log_pxz)
            tf.summary.scalar(record_string + 'temp', _temp)

        return _mean, elbo

    def call_sample(self, batch_shape=None, z=None):
        """

        Args:
            batch_shape:
            z:

        Returns:

        """
        if batch_shape is None and z is None:
            raise NotImplementedError('either a batch_shape or some exact samples of z are required')

        if z is None:
            assert batch_shape[-1] == self.d_model
            logit = tf.math.log(tf.ones(batch_shape) / float(self.d_model))
            logit = tf.reshape(logit, [-1, self.d_model])

            z = tf.random.categorical(logit, 1)  # [prod(...) 1]
            z = tf.cast(tf.reshape(z, batch_shape[:-1]), tf.int32)
            z = tf.one_hot(z, depth=self.d_model, dtype=tf.float32, axis=-1)

        likelihood_parameter = self.decoder(z, training=False)

        return likelihood_parameter if self.likelihood == 'Bernoulli' else likelihood_parameter[0]
