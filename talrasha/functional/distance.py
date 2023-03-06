from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from functools import partial
from typing import Optional, Callable

OVERFLOW_MARGIN = 1e-8


@tf.function
def adjacency_euclidean(x: tf.Tensor, y: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    computing pairwise SQUARED distances between any two ROW elements from x and y.

    This is quite useful for attentions, Gaussian kernels, GCNs and MMD computation.

    Args:
        x: [N D]
        y: [M D]

    Returns:
        a tensor of shape [N M]
    """
    if y is not None:

        assert x.shape.__len__() == 2
        assert y.shape.__len__() == 2
        assert x.shape[1] == y.shape[1]
    else:
        y = x

    xx = tf.reduce_sum(x * x, axis=1)[:, tf.newaxis]
    yy = tf.reduce_sum(y * y, axis=1)[:, tf.newaxis]
    xy = tf.matmul(x, y, transpose_b=True)

    return xx - 2 * xy + yy


@tf.function
def adjacency_dot(x: tf.Tensor, y: Optional[tf.Tensor] = None, normalize=False) -> tf.Tensor:
    """
    computing pairwise dot product between any two ROW elements from x and y.

    This is quite useful for attentions and contrastive learning computation.

    Args:
        x: [N D]
        y: [M D]
        normalize: if l2 normalization is applied to the last axis

    Returns:
        a tensor of shape [N M]
    """
    if y is not None:
        shape_1 = tf.shape(x)
        shape_2 = tf.shape(y)
        assert shape_1.__len__() == 2
        assert shape_2.__len__() == 2
        assert shape_1[1] == shape_2[1]
    else:
        y = x

    if normalize:
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)

    return tf.matmul(x, y, transpose_b=True)


@tf.function
def kernel_inverse_multiquadratic(x, y, c=None) -> tf.Tensor:
    """
    Computing pairwise results of c/(c+|x-y|^2)

    This is very useful in MMD and WAE.

    Args:
        x: [N D]
        y: [M D]
        c: the hyperparameter, by default, it will be d_model * 2.

    Returns:
        a tensor of shape [N M]
    """

    if c is None:
        c = tf.cast(tf.shape(x)[-1] * 2, tf.float32)

    return c / (c + adjacency_euclidean(x, y))


def mmd(x, y, rkhs='imq', sqrt=False):
    """
    MMD!

    Args:
        x: [N ... D]
        y: [M ... D]
        rkhs: The kernel. Only 'imq' is supported atm.
        sqrt: If sqrt is applied. When used together with squared l2 losses, it would be better to have squared MMD,

    Returns:
        a single value of discrepancy.
    """
    d_model = tf.shape(x)[-1]
    if rkhs == 'imq':
        kernel: Callable = partial(kernel_inverse_multiquadratic, c=tf.cast(d_model * 2, tf.float32))

    else:
        raise NotImplementedError('No kernel supported')

    x = tf.reshape(x, [-1, d_model])
    y = tf.reshape(y, [-1, d_model])

    xx = tf.reduce_mean(kernel(x, x))
    xy = tf.reduce_mean(kernel(x, y))
    yy = tf.reduce_mean(kernel(y, y))

    rslt = xx - 2 * xy + yy

    return tf.sqrt(tf.maximum(rslt, 0.)) if sqrt else rslt
