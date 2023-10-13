from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Union, Iterable
import numpy as np

from functools import partial

if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


class DiffusionModel(keras.Model):
    """
    This is implementing the (somewhat) vanilla diffusion model from the paper
    * [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

    We choose to implement the version of which the neural network reconstructs epsilon from x_t and the variations are
    untrainable.

    This module only includes the explicit DDPM sampler.

    """
    def __init__(self,
                 total_step,
                 beta: Union[Iterable, tf.Tensor, None] = None,
                 backbone: Union[keras.layers.Layer, keras.Model, None] = None,
                 sigma_type: str = 'small',
                 *args, **kwargs):
        """

        :param total_step: t
        :param beta: a list of betas. Length should be equal to `total_step`
        :param backbone: The network. We give an implementation example of a default UNET with pre-normed attention.
        :param sigma_type: 'small' for sigma = beta and 'large' for sigma = the complex version
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.total_step = total_step
        self.rng = tf.random.Generator.from_seed(2)
        self.backbone = backbone
        self.sigma_type = sigma_type
        beta = np.linspace(1e-4, 0.02, total_step) if beta is None else beta
        self._alpha_beta(beta)

    def _alpha_beta(self, beta):
        beta_64 = tf.constant(beta)
        beta_64 = tf.cast(beta_64, dtype=tf.float64)
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
    def _get_value(source, t, num_dim: int = 1):
        """

        :param source: [T]
        :param t: [N]
        :param num_dim: Number of dimensions of the data, e.g., 1 for vectors and 3 for
                image feature maps.
        :return: the values with shape [N, *[1]*num_dim]
        """
        v = tf.gather(source, indices=t)
        for _ in range(num_dim):
            v = v[..., tf.newaxis]

        return tf.stop_gradient(v)

    def train(self, x, t=None, training=True):
        """

        Computing the term ||eps - network(sqrt_alpha_bar * x0 - sqrt_1_alpha_bar * eps, t)||, a weighted VLB of the
        reverse process.

        :param x: [N ... D]
        :param t: [N]
        :param training:
        :return: the loss
        """
        eps = self.rng.normal(tf.shape(x))  # [N ... D]
        batch_alpha_bar = self._get_value(self.alpha_bar, t, num_dim=len(tf.shape(x)) - 1)  # [N ... 1]
        batch_one_minus = 1. - batch_alpha_bar

        a1 = tf.sqrt(tf.maximum(batch_alpha_bar, 0))
        a2 = tf.sqrt(tf.maximum(batch_one_minus, 0))

        net_input = a1 * x + a2 * eps

        x_pred = self.backbone(net_input, t, training=training)

        vlb = tf.reduce_sum(tf.pow(eps - x_pred, 2), axis=list(range(1, len(x.shape))))
        return tf.reduce_mean(vlb)

    def sample_single_step(self, x_t, t: int):
        """
        This is to sample x_{t-1} using the eps prediction backbone and x_t

        :param x_t:
        :param t: scalar
        :return:
        """
        batch_shape = tf.shape(x_t)
        z = self.rng.normal(batch_shape) * float(t != 0)
        _t = tf.fill([batch_shape[0]], t)

        # From here on we are computing x_prev = k0 * (xt - k1 * backbone(xt, t)) + k2 * z

        alpha = self._get_value(self.alpha, _t, num_dim=len(batch_shape) - 1)
        k_0 = 1 / tf.sqrt(alpha)

        alpha_bar = self._get_value(self.alpha_bar, t, num_dim=len(batch_shape) - 1)
        beta = 1 - alpha
        sqrt_1_alpha_bar = tf.sqrt(tf.maximum(1e-9, 1 - alpha_bar)) + 1e-9
        k_1 = beta / sqrt_1_alpha_bar

        log_sigma_sqr = self._get_value(self.log_sigma_sqr, _t, num_dim=len(batch_shape) - 1)
        k_2 = tf.exp(log_sigma_sqr * .5)

        eps = self.backbone(x_t, _t, training=False)

        x_prev = k_0 * (x_t - k_1 * eps) + k_2 * z

        return x_prev

    def sample_single_step_clipped(self, x_t, t: int):
        """
        This is also sampling x_{t-1} using the eps prediction backbone and x_t

        However, since an intermediate step predicting x_0 could be followed by value clipping, the
        overall implementation would be slightly different. In theory, this method is producing the
        same thing as sample_single_step.

        :param x_t:
        :param t: scalar
        :return:
        """
        batch_shape = tf.shape(x_t)
        z = self.rng.normal(batch_shape)
        _t = tf.ones([batch_shape[0]], dtype=tf.int32) * t

        eps = self.backbone(x_t, _t, training=False)
        data_dim = len(batch_shape) - 1

        take = partial(self._get_value, t=_t, num_dim=data_dim)

        alpha = take(self.alpha)
        alpha_bar = take(self.alpha_bar)
        k_0 = 1. / tf.sqrt(alpha_bar)
        k_1 = tf.sqrt((1. - alpha_bar) / alpha_bar)
        x_0 = k_0 * x_t - k_1 * eps

        x_0 = tf.clip_by_value(x_0, -1, 1)

        beta = take(self.beta)
        shifted = take(self.shifted)
        log_sigma_sqr = take(self.log_sigma_sqr)
        k_2 = beta * tf.sqrt(shifted) / (1. - alpha_bar + 1e-9)
        k_3 = (1 - shifted) * tf.sqrt(alpha) / (1. - alpha_bar + 1e-9)
        k_4 = tf.exp(log_sigma_sqr * .5)

        x_prev = k_2 * x_0 + k_3 * x_t + k_4 * z

        return x_prev

    def sample(self, x):
        rslt = [x]
        for t in range(self.total_step - 1, -1, -1):
            x = self.sample_single_step_clipped(x, t)

            if t == self.total_step // 2:
                rslt.append(tf.identity(x))

        rslt.append(x)

        return rslt

    def call(self, x, training=True, mask=None, step=-1):
        if training:
            batch_size = tf.shape(x)[0]
            t = self.rng.uniform(shape=[batch_size], minval=0, maxval=self.total_step - 1, dtype=tf.int32)

            vlb = self.train(x, t, training=training)
            if step >= 0:
                tf.summary.scalar('train/vlb', vlb, step=step)
            return vlb
        else:
            return self.sample(x)


if __name__ == '__main__':
    from talrasha.model.diffusion._helpers import _UNet

    backbone = _UNet()
    model = DiffusionModel(50, backbone=backbone)
    x = tf.zeros([2, 32, 32, 3])
    l = model(x)
    gen = model(x, training=False)
    print(l.numpy(), gen[-1].numpy())
