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


class _DefaultMNISTNet(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fc_t = keras.layers.Dense(512, activation=tf.nn.swish)


class BasicDiffusion(keras.Model):
    """
    The basic Gaussian diffusion model
    """

    def __init__(self, total_step, beta: Optional[Iterable, tf.Tensor] = None,
                 backbone: Optional[keras.layers.Layer, keras.Model] = None,
                 *args, **kwargs):
        """

        :param total_step:
        :param beta:
        :param backbone:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.beta = np.linspace(1e-4, 0.02, total_step) if beta is None else beta
        self.total_step = total_step
