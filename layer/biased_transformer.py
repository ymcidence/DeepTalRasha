from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from tensorflow import keras
import typing
import numpy as np

# as per the IDE resolving bug
if typing.TYPE_CHECKING:
    # noinspection PyProtectedMember
    from keras.api._v2 import keras


def dot_group_bias(q_group, k_group, group_emb):
    """

    :param q_group: [N H L G]
    :param k_group: [N H L G]
    :param group_emb: [G D]
    :return:
    """
    q = tf.einsum('nhlg,gd->nhld', q_group, group_emb)
    k = tf.einsum('nhlg,gd->nhld', k_group, group_emb)

    bias = tf.einsum('nhld,nhsd->nhls', q, k)

    return bias
