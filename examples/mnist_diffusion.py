from __future__ import absolute_import, print_function, division, unicode_literals

import os
import talrasha as trs
import tensorflow as tf
from tensorflow import keras
import typing
from typing import List
import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras

ROOT_PATH = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]


def epoch(model: keras.Model, data: List[tf.data.Dataset], opt: keras.optimizers.Optimizer,
          writer: tf.summary.SummaryWriter):
    return 0


def main():
    model = trs.model.BasicDiffusion(100)
    mnist = trs.util.get_toy_data('mnist', 32)
    opt = keras.optimizers.Adam(2e-4)
    writer = trs.util.make_training_folder(ROOT_PATH, 'mnist_diffusion')
