from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
from typing import Union, Iterable

if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


class SiLU(keras.Model):
    def __init__(self, *args, **kwargs):
        """
        x * sigmoid(x)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return inputs * tf.nn.sigmoid(inputs)
