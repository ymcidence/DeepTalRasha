from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

OVERFLOW_MARGIN = 1e-8


def _get_sample_tensor_shape(old_shape: tf.Tensor, sample_size: int) -> tf.Tensor:
    new_shape = tf.concat([old_shape[0: 1], [sample_size], old_shape[1:]], axis=0)
    return new_shape


class GaussianReparameterization(keras.layers.Layer):
    def __init__(self, d_model: int, **kwargs):
        """

        Args:
            d_model: Dim of the latent space
            **kwargs:
        """
        super().__init__(**kwargs)
        self.d_model = d_model

    # noinspection PyMethodOverriding
    def call(self, mean, log_var, sample_size=1, stochastic=True, *args, **kwargs):
        """

        Args:
            mean:
            log_var:
            sample_size: how many random samples to take for Monte Carlo gradient estimation at each step
            stochastic:
            *args:
            **kwargs:

        Returns:
            tensor of shape [N sample_size ... D] if training==True or [N ... D] for training==False
        """
        batch_shape = tf.shape(mean)
        assert batch_shape[-1] == self.d_model

        new_shape = _get_sample_tensor_shape(batch_shape, sample_size)

        if stochastic:
            mean = mean[:, tf.newaxis, ...]
            log_var = log_var[:, tf.newaxis, ...]
            eps = tf.random.normal(new_shape)
        else:
            eps = tf.zeros_like(log_var, dtype=tf.float32)

        sample = mean + tf.exp(.5 * log_var) * eps

        return sample

    # def white_sample(self, batch_size):
    #     batch_shape = [batch_size, self.d_model]
    #     return tf.random.normal(batch_shape)


class GumbelReparameterization(keras.layers.Layer):
    def __init__(self, d_model: int, temp: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp
        self.d_model = d_model
        self.rng = tf.random.Generator.from_seed(0)

    def call(self, logit: tf.Tensor, sample_size=1, stochastic=True, temp=None, *args, **kwargs):
        """
        the Gumbel trick
        Args:
            logit: [N ... D]
            sample_size: how many random samples to take for Monte Carlo gradient estimation at each step
            stochastic: if stochasticity applies. Use True for training.
            temp:
            *args:
            **kwargs:

        Returns:

        """
        batch_shape = tf.shape(logit)
        assert batch_shape[-1] == self.d_model
        temp = temp if temp is not None else self.temp

        new_shape = _get_sample_tensor_shape(batch_shape, sample_size)

        if stochastic:
            logit = logit[:, tf.newaxis, ...]
            # Be very careful with tf.random.uniform as it produces CONSTANT results on some tf versions!
            u = self.rng.uniform(new_shape, minval=0, maxval=1)
            gumbel_sample = -tf.math.log(-tf.math.log(u + OVERFLOW_MARGIN) + OVERFLOW_MARGIN)
            new_logit = logit + gumbel_sample

            return tf.nn.softmax(new_logit / temp, axis=-1)
        else:
            # return tf.nn.softmax(logit / temp, axis=-1)

            argmax = tf.argmax(logit, axis=-1)
            return tf.cast(tf.one_hot(argmax, axis=-1, depth=self.d_model), tf.float32)
