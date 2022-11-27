from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_toy_data(set_name: str, batch_size: int, data_dir=None, shuffle_buffer=30000, map_function=None) -> List[
    tf.data.Dataset]:
    """

    Args:
        set_name: mnist or cifar
        batch_size:
        data_dir: To identify where the downloaded files would be stored.
        shuffle_buffer: shuffle buffer size
        map_function: A map function, default to flattening + [0, 1] normalization

    Returns:

    """

    def _map(x):
        x['feat'] = tf.cast(tf.reshape(x['image'], [-1]), dtype=tf.float32) / 255.
        return x

    data = tfds.load(set_name, data_dir=data_dir)

    mf = map_function if map_function is not None else _map

    data_train = data['train'].shuffle(shuffle_buffer).map(mf, num_parallel_calls=AUTOTUNE).batch(
        batch_size).prefetch(AUTOTUNE)
    data_test = data['test'].map(mf, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return [data_train, data_test]
