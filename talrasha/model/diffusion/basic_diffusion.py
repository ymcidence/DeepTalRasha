from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Union, Iterable
import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

from talrasha.functional.positional_emb import sinusoidal_encoding


class _DefaultMNISTNet(keras.layers.Layer):
    """
    A diffusion-only MNIST network, modelling p(x(t-1)|x(t)) or literally eps(x_t, t)

    """

    def __init__(self, total_step, **kwargs):
        """

        Args:
            total_step: The total steps of a diffusion model's forward/reverse trajectory
            **kwargs:
        """
        super().__init__(**kwargs)
        self.total_step = total_step

        self.fc_t = keras.layers.Dense(512, activation=tf.nn.swish)
        self.fc_i = keras.layers.Dense(512, activation=tf.nn.swish)

        self.net = keras.Sequential([
            keras.layers.LayerNormalization(),
            keras.layers.Dense(28 * 28, activation=tf.nn.swish),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(28 * 28)
        ])

    # noinspection PyMethodOverriding
    def call(self, x, t, training=True, *args, **kwargs):
        """

        Args:
            x: [N D]
            t: [N]
            training:
            *args:
            **kwargs:

        Returns:

        """
        x = self.fc_i(x, training=training)
        s = sinusoidal_encoding(t, 512, total_step=self.total_step)
        s = self.fc_t(s, training=training)

        mixed = x + s

        rslt = self.net(mixed, training=True)

        return rslt


class BasicDiffusion(keras.Model):
    """
    The basic Gaussian diffusion model
    """

    def __init__(self, total_step, beta: Union[Iterable, tf.Tensor, None] = None,
                 backbone: Union[keras.layers.Layer, keras.Model, None] = None,
                 *args, **kwargs):
        """

        Args:
            total_step: The total steps of a diffusion model's reverse trajectory
            beta: The diffusion rates defined by the original paper, a linspace by default.
            backbone: The callable eps_theta, a MNIST MLP by default.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.total_step = total_step

        b = np.linspace(1e-4, 0.02, total_step) if beta is None else beta
        self._alpha_beta(b)

        self.backbone = _DefaultMNISTNet(total_step=total_step) if backbone is None else backbone

    def _alpha_beta(self, beta):
        self.beta = tf.cast(beta, tf.float32)
        self.alpha = 1 - self.beta

        repeat_alpha = tf.tile(self.alpha[tf.newaxis, :], [self.total_step, 1])
        repeat_alpha = tf.linalg.band_part(repeat_alpha, -1, 0)

        ones = tf.ones([self.total_step, self.total_step])
        eye = tf.eye(self.total_step)

        mask = tf.linalg.band_part(ones, 0, -1) - eye

        self.alpha_bar = tf.reduce_prod(repeat_alpha + mask, axis=-1)

        shifted = tf.concat([tf.constant([1], tf.float32), self.alpha_bar[:-1]], axis=0)
        ratio = (1 - shifted) / (1 - self.alpha_bar)

        sigma_sqr = ratio * self.beta
        self.sigma = tf.sqrt(sigma_sqr)

    def call_train(self, x, t):
        """


        Args:
            x: x_0 of a diffusion model. [N D]
            t: The (sampled) step counts. Can be different for each datum. [N]

        Returns:
            An ELBO estimation of a sampled step t
        """

        eps = tf.random.normal(tf.shape(x))  # [N D]

        batch_alpha_bar = tf.sqrt(tf.gather(self.alpha_bar, indices=t))  # [N]
        a1 = tf.sqrt(batch_alpha_bar)[:, tf.newaxis]  # [N 1]
        a2 = tf.sqrt(1 - batch_alpha_bar)[:, tf.newaxis]  # [N 1]

        x_in = a1 * x + a2 * eps
        eps_theta = self.backbone(x_in, t, training=True)  # [N D]

        elbo = tf.reduce_sum(tf.pow(eps - eps_theta, 2), axis=-1)

        return tf.reduce_mean(elbo)

    def call_sample(self, x):
        batch_shape = tf.shape(x)
        batch_size = batch_shape[0]
        rslt = [x]
        for i in range(self.total_step - 1, -1, -1):
            z = tf.random.normal(batch_shape)
            t = tf.ones([batch_size], dtype=tf.int32) * i
            sigma = self.sigma[i]
            alpha = self.alpha[i]
            alpha_bar = self.alpha_bar[i]
            ratio = (1 - alpha_bar) / tf.sqrt(1 - alpha_bar)
            x = 1 / tf.sqrt(alpha) * (x - ratio * self.backbone(x, t, training=False)) + sigma * z

            if i == self.total_step // 2:
                rslt.append(tf.identity(x))

        rslt.append(x)

        return rslt

    # noinspection PyMethodOverriding
    def call(self, inputs, training=True, step=-1):
        if training:
            batch_size = tf.shape(inputs)[0]
            t = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.total_step - 1, dtype=tf.int32)
            elbo = self.call_train(inputs, t)
            if step >= 0:
                tf.summary.scalar('train/elbo', elbo, step=step)
            return elbo
        else:
            return self.call_sample(inputs)

# if __name__ == '__main__':
#     m = BasicDiffusion(100)
#     print(m.alpha_bar)
