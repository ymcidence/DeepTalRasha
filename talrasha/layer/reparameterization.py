from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


class GaussianReparameterization(keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
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

        new_shape = tf.concat([batch_shape[0: 1], [sample_size], batch_shape[1:]], axis=0)

        if stochastic:
            mean = mean[:, tf.newaxis, ...]
            log_var = log_var[:, tf.newaxis, ...]
            eps = tf.random.normal(new_shape)
        else:
            eps = tf.zeros_like(log_var, dtype=tf.float32)

        sample = mean + tf.exp(.5 * log_var) * eps

        return sample
