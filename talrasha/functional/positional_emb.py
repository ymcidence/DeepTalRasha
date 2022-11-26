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


def sinusoidal_encoding(position: Union[int, np.ndarray, list], d_model, total_step=10000):
    """
    The sinusoidal positional embedding function used in the original transformer paper.

    Note that this gives a 2-D array.

    Args:
        position: Literally the position t.
        d_model: Channel number.
        total_step: The total length a sequence could be.
            Changing it may influence the 'resolution' of the codes.

    Returns:
        The codes of the input positions
    """

    def get_angles(pos, i):
        angle_rates = 1 / np.power(total_step, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    if isinstance(position, int):
        p = np.arange(position)
    elif isinstance(position, list) or isinstance(position, np.ndarray):
        p = np.asarray(position)
    else:
        raise NotImplementedError('The input type of position is not supported')

    p = p[..., np.newaxis]
    angle_rads = get_angles(p, np.arange(d_model)[np.newaxis, :])

    #
    angle_rads[..., 0::2] = np.sin(angle_rads[..., 0::2])

    #
    angle_rads[..., 1::2] = np.cos(angle_rads[..., 1::2])

    pos_encoding = angle_rads  # [np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':
    x = np.arange(10)[np.newaxis, :]
    x = np.concatenate([x, x // 4], axis=0)
    a = sinusoidal_encoding(x, 64)
    print(a)
