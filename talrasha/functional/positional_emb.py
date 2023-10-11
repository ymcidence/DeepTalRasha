from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from typing import Union, Iterable
import numpy as np


def sinusoidal_encoding(position: Union[int, Iterable], d_model, total_step=10000) -> tf.Tensor:
    """
    The sinusoidal positional embedding function used in the original transformer paper.

    Args:
        position: Literally the positions.
            -- A single scalar refers to the linspace positions upto that value.
            -- A tensor-like input describes the actual indices.
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
    elif isinstance(position, tf.Tensor):
        return tf_sinusoidal_encoding(position, d_model, total_step)
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


def tf_sinusoidal_encoding(position: tf.Tensor, d_model, total_step=10000) -> tf.Tensor:
    def get_angles(pos, i):
        angle_rates = 1 / tf.pow(np.float32(total_step), (2 * (i // 2)) / np.float32(d_model))
        return tf.cast(pos, tf.float32) * tf.cast(angle_rates, tf.float32)

    p = position[..., tf.newaxis]

    even_rad = get_angles(p, tf.range(0, d_model, 2, dtype=tf.float32)[tf.newaxis, :])
    odd_rad = get_angles(p, tf.range(1, d_model, 2, dtype=tf.float32)[tf.newaxis, :])

    even_rad = tf.sin(even_rad)
    odd_rad = tf.cos(odd_rad)
    rad = tf.concat([even_rad, odd_rad], axis=-1)

    # staggering dimensions
    half_d = int(d_model / 2)
    perm = tf.reshape(tf.range(d_model), [2, half_d])
    perm = tf.reshape(tf.transpose(perm, [1, 0]), [d_model])

    return tf.gather(rad, perm, axis=-1)

# if __name__ == '__main__':
#     x = np.arange(10)[np.newaxis, :]
#     x = np.concatenate([x, x // 4], axis=0)
#     x1 = tf.cast(x, dtype=tf.int32)
#     a = sinusoidal_encoding(x, 64)
#     a1 = sinusoidal_encoding(x1, 64)
#     print(a, a1)
