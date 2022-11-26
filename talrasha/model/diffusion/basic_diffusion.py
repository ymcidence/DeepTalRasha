from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Optional, Iterable
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
            total_step: The total steps of a diffusion model's reverse trajectory
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
            x:
            t:
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

    def __init__(self, total_step, beta: Optional[Iterable, tf.Tensor] = None,
                 backbone: Optional[keras.layers.Layer, keras.Model] = None,
                 *args, **kwargs):
        """

        Args:
            total_step: The total steps of a diffusion model's reverse trajectory
            beta:
            backbone:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.beta = np.linspace(1e-4, 0.02, total_step) if beta is None else beta
        self.total_step = total_step
        self.backbone = _DefaultMNISTNet(total_step=total_step) if backbone is None else backbone
