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

        self.fc_t = keras.Sequential([
            keras.layers.Dense(28 * 28),
            keras.layers.LeakyReLU(.1)
        ])
        self.fc_i = keras.Sequential([
            keras.layers.Dense(28 * 28),
            keras.layers.LeakyReLU(.1)
        ])

        self.net = keras.Sequential([
            self.fc(2048),
            self.fc(2048),
            self.fc(2048),
            keras.layers.Dense(28 * 28)
        ])

    @staticmethod
    def fc(d_model):
        return keras.Sequential([
            keras.layers.Dense(d_model),
            # keras.layers.LayerNormalization(),
            keras.layers.LeakyReLU(.1),
            keras.layers.Dropout(.8)
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
        s = sinusoidal_encoding(t, 512)
        s = self.fc_t(s, training=training)

        x_0 = tf.concat([x, s], axis=-1)
        rslt = self.net(x_0, training=training)

        return rslt


class BasicDiffusion(keras.Model):
    """
    The basic Gaussian diffusion model
    """

    def __init__(self, total_step, beta: Union[Iterable, tf.Tensor, None] = None,
                 backbone: Union[keras.layers.Layer, keras.Model, None] = None,
                 sigma_type: str = 'small',
                 *args, **kwargs):
        """

        Args:
            total_step: The total steps of a diffusion model's reverse trajectory
            beta: The diffusion rates defined by the original paper, a linspace by default.
            backbone: The callable eps_theta, a MNIST MLP by default.
            sigma_type: 'large' (the sophisticated one) or 'small' (sigma=sqrt(beta))
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)

        self.total_step = total_step

        b = np.linspace(1e-4, 0.02, total_step) if beta is None else beta
        self.sigma_type = sigma_type
        self._alpha_beta(b)

        self.backbone = _DefaultMNISTNet(total_step=total_step) if backbone is None else backbone

    def _alpha_beta(self, beta):
        beta_64 = tf.cast(beta, tf.float64)
        alpha_64 = 1 - beta_64

        alpha_bar_64 = tf.math.cumprod(alpha_64)

        shifted_64 = tf.concat([tf.constant([1], tf.float64), alpha_bar_64[:-1]], axis=0)
        ratio_64 = (1 - shifted_64) / (1 - alpha_bar_64)

        sigma_sqr_64 = ratio_64 * beta_64

        self.beta = tf.cast(beta, tf.float32)
        self.alpha = tf.cast(alpha_64, tf.float32)
        self.alpha_bar = tf.cast(alpha_bar_64, tf.float32)
        self.shifted = tf.cast(shifted_64, tf.float32)
        if self.sigma_type == 'large':
            self.sigma = tf.cast(tf.sqrt(sigma_sqr_64), tf.float32)

            log_sigma_sqr_64 = np.log(np.append(sigma_sqr_64[1], sigma_sqr_64[1:]))

        elif self.sigma_type == 'small':
            self.sigma = tf.cast(tf.sqrt(beta_64), tf.float32)
            log_sigma_sqr_64 = np.log(np.append(sigma_sqr_64[1], beta_64[1:]))
        else:
            raise NotImplementedError('variance type not supported')

        self.log_sigma_sqr = tf.cast(log_sigma_sqr_64, tf.float32)

    @staticmethod
    def _get_value(source, t, data_dim: int = 1):
        """
        Get the coefficients from source indexed by t

        Args:
            source: [T]
            t: [N]
            data_dim: The dimensionality of the data, e.g., 1 for vectors and 3 for
                image feature maps.

        Returns:
            the values with shape [N, 1... of data_dim]
        """
        v = tf.gather(source, indices=t)
        for _ in range(data_dim):
            v = v[..., tf.newaxis]

        return v

    def call_train(self, x, t):
        """


        Args:
            x: x_0 of a diffusion model. [N D]
            t: The (sampled) step counts. Can be different for each datum. [N]

        Returns:
            An ELBO estimation of a sampled step t
        """

        eps = tf.random.normal(tf.shape(x))  # [N D]

        batch_alpha_bar = self._get_value(self.alpha_bar, t, data_dim=len(tf.shape(x)) - 1)  # [N]
        a1 = tf.sqrt(batch_alpha_bar)
        a2 = tf.sqrt(1 - batch_alpha_bar)

        x_in = a1 * x + a2 * eps
        eps_theta = self.backbone(x_in, t, training=True)  # [N D]

        vlb = tf.reduce_sum(tf.pow(eps - eps_theta, 2), axis=list(range(1, len(x.shape))))

        return tf.reduce_mean(vlb)

    def single_sample(self, x, t, batch_shape):
        z = tf.random.normal(batch_shape)
        t = tf.ones([batch_shape[0]], dtype=tf.int32) * t

        eps_pred = self.backbone(x, t, training=False)
        data_dim = len(batch_shape) - 1

        alpha_bar = self._get_value(self.alpha_bar, t, data_dim=data_dim)
        shifted = self._get_value(self.shifted, t, data_dim=data_dim)
        beta = self._get_value(self.beta, t, data_dim=data_dim)
        log_sigma_sqr = self._get_value(self.log_sigma_sqr, t, data_dim=data_dim)

        sigma = tf.exp(log_sigma_sqr * .5)
        mask = tf.cast(tf.equal(t, 0), tf.float32)
        mask = tf.reshape(mask, [batch_shape[0]] + [1] * data_dim)

        x_0_pred = x / tf.sqrt(alpha_bar) - eps_pred * tf.sqrt(1. / alpha_bar - 1.)
        x_0_pred = tf.clip_by_value(x_0_pred, -1, 1)

        x_mean = x_0_pred * beta * tf.sqrt(shifted) / (1 - alpha_bar) + tf.sqrt(1 - beta) * x * (1 - shifted) / (
                1 - alpha_bar)

        x = x_mean + mask * sigma * z

        return x, sigma

    def call_sample(self, x):
        batch_shape = tf.shape(x)
        rslt = [x]
        for i in range(self.total_step - 1, -1, -1):
            x, sigma = self.single_sample(x, i, batch_shape)
            if i == self.total_step // 2:
                rslt.append(tf.identity(x))

        rslt.append(x)

        return rslt

    # noinspection PyMethodOverriding
    def call(self, inputs, training=True, step=-1):
        if training:
            batch_size = tf.shape(inputs)[0]
            t = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.total_step - 1, dtype=tf.int32)
            vlb = self.call_train(inputs, t)
            if step >= 0:
                tf.summary.scalar('train/vlb', vlb, step=step)
            return vlb
        else:
            return self.call_sample(inputs)

# if __name__ == '__main__':
#     m = BasicDiffusion(100)
#     print(m.alpha_bar)
