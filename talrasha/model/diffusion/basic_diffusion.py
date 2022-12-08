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


class _ResBlock(keras.layers.Layer):
    def __init__(self, channel, kernel=3, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.conv_1 = keras.Sequential([
            keras.layers.Conv2D(channel, kernel_size=kernel, strides=stride, padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ])
        self.fc = keras.Sequential([
            keras.layers.ReLU(),
            keras.layers.Dense(channel),
            keras.layers.BatchNormalization()
        ])
        self.conv_2 = keras.Sequential([
            keras.layers.Dropout(.1),
            keras.layers.Conv2D(channel, kernel_size=kernel, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ])
        self.shortcut = keras.Sequential([
            keras.layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ])

    # noinspection PyMethodOverriding
    def call(self, x, t, training=True, *args, **kwargs):
        h = self.conv_1(x, training=training)
        z = self.fc(t, training=training)[:, tf.newaxis, tf.newaxis, :]
        hz = tf.nn.relu(h + z)
        h = self.conv_2(hz, training=training)
        s = self.shortcut(x, training=training)
        return tf.nn.relu(h + s)


class _UpscaleResBlock(keras.layers.Layer):
    def __init__(self, channel, kernel=3, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.upscale = keras.layers.UpSampling2D((2, 2), data_format='channels_last')
        self.res_block = _ResBlock(channel, kernel=kernel, stride=stride)

    # noinspection PyMethodOverriding
    def call(self, x, h, t, training=True, *args, **kwargs):
        x = self.upscale(x)
        z = tf.concat([x, h], axis=-1)
        return self.res_block(z, t, training=training)


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

        self.encoder_0 = keras.Sequential([
            keras.layers.Conv2D(16, 3, padding='same')
        ])
        self.fc_t = keras.layers.Dense(512)
        self.fc_0 = keras.Sequential([
            keras.layers.ReLU(),
            keras.layers.Dense(16)
        ])

        self.encoder_1 = _ResBlock(32, stride=2)
        self.encoder_2 = _ResBlock(64, stride=2)
        self.decoder_1 = _UpscaleResBlock(32)

        self.decoder_0 = _UpscaleResBlock(16)

        self.final = keras.layers.Conv2D(1, 3, padding='same')

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
        e_0 = self.encoder_0(x, training=training)
        s = sinusoidal_encoding(t, 128)
        s = self.fc_t(s, training=training)

        s_0 = self.fc_0(s, training=training)[:, tf.newaxis, tf.newaxis, :]
        e_0 = tf.nn.relu(s_0 + e_0)

        e_1 = self.encoder_1(e_0, s, training=training)  # [14 14]
        e_2 = self.encoder_2(e_1, s, training=training)  # [7 7]

        d_1 = self.decoder_1(e_2, e_1, s, training=training)  # [14 14]
        d_0 = self.decoder_0(d_1, e_0, s, training=training)  # [28 28]

        return self.final(d_0, training=training)


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


if __name__ == '__main__':
    m = _DefaultMNISTNet(100)
    img = tf.zeros([1, 28, 28, 1], dtype=tf.float32)
    t = tf.constant([4], dtype=tf.int32)

    a = m(img, t)

    print(a.shape)
