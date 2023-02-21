from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from typing import Optional

OVERFLOW_MARGIN = 1e-8


def gaussian_kld(mean_1: tf.Tensor, log_var_1: tf.Tensor, mean_2: Optional[tf.Tensor] = None,
                 log_var_2: Optional[tf.Tensor] = None, reduce=False) -> tf.Tensor:
    """
    The Kullback-Leibler divergence between 2 Gaussians using the Rao-Blackwellized fashion

    Note that the variances MUST be diagonal matrices, otherwise, I don't know how to compute...

    P1 = N(mean_1, diag(var_1))
    P2 = N(mean_2, diag(var_2))

    Args:
        mean_1: [N ... D], the mean vector of the first distribution
        log_var_1: [N ... D], the logarithm of the variance diagonal of the first distribution
        mean_2: [N ... D], the mean vector of the 2nd distribution, leaving None for a standard N(0, I)
        log_var_2: [N ... D], the logarithm of the variance diagonal of the 2nd distribution,
                leaving None for a standard N(0, I)
        reduce: if reduce_mean is applied on the first axis

    Returns:
        the KLDs averaged throughout the batch axis if reduce=True, otherwise the KLDs of each batch entry.
    """

    if mean_2 is None:
        # -0.5 * sum(1 + log_var - mean^2 - var)
        kl = -.5 * tf.reduce_sum(log_var_1 - tf.square(mean_1) - tf.exp(log_var_1) + 1,
                                 axis=list(range(1, len(mean_1.shape))))

    else:
        # see https://statproofbook.github.io/P/mvn-kl.html
        batch_size = tf.shape(mean_1)[0]

        mean_1 = tf.reshape(mean_1, [batch_size, -1])
        mean_2 = tf.reshape(mean_2, [batch_size, -1])
        log_var_1 = tf.reshape(log_var_1, [batch_size, -1])
        log_var_2 = tf.reshape(log_var_2, [batch_size, -1])

        var_1 = tf.exp(log_var_1)
        inverse_var_2 = tf.exp(-log_var_2)  # [N D']
        log_det_var_1 = tf.reduce_sum(log_var_1, axis=-1)  # [N]
        log_det_var_2 = tf.reduce_sum(log_var_2, axis=-1)  # [N]
        trace_vars = tf.reduce_sum(inverse_var_2 * var_1, axis=-1)  # [N]
        dif_means = mean_2 - mean_1  # [N D']
        corr_means = tf.einsum('nd,nd->n', dif_means * inverse_var_2, dif_means)
        d = tf.cast(tf.shape(mean_1)[-1], tf.float32)

        kl = 0.5 * (corr_means + trace_vars - log_det_var_1 + log_det_var_2 - d)
    return tf.reduce_mean(kl) if reduce else kl


def gaussian_prob(value: tf.Tensor, mean: tf.Tensor, log_var: tf.Tensor, logarithm=True, reduce=False):
    """
    To compute the probability of P(value) = N(value| mean, var)

    Args:
        value: [N ... D]
        mean: [N ... D] or [N L ... D], the Gaussian mean of P()
        log_var: [N ... D] or [N L ... D], the diagonal of the Gaussian variance of p()
        logarithm: if True, returns log P(value), otherwise P(value)
        reduce: if reduce_mean is applied on the first axis

    Returns:
        The Gaussian probs of the values. log and reduce may apply depending on logarithm and reduce.
    """

    log_pi_sqr = tf.math.log(np.pi * 2)

    # multiple means and vars usually come from the sampling process of the reparameterization trick
    # in this case, we need to compute the expectation of the likelihoods
    condition = tf.shape(value).__len__() < tf.shape(mean).__len__()
    if condition:
        value = value[:, tf.newaxis, ...]
        reduce_start = 2
    else:
        reduce_start = 1

    rslt = -.5 * (tf.square(value - mean) * tf.exp(-log_var) + log_var + log_pi_sqr)

    # noinspection DuplicatedCode
    rslt = tf.reduce_sum(rslt, axis=list(range(reduce_start, len(rslt.shape))))

    # multiple means and vars usually come from the sampling process of the reparameterization trick
    # in this case, we need to compute the expectation of the likelihoods
    if condition:
        rslt = tf.reduce_mean(rslt, axis=-1)

    if not logarithm:
        rslt = tf.exp(rslt)

    return tf.reduce_mean(rslt) if reduce else rslt


def bernoulli_prob(value: tf.Tensor, prob: tf.Tensor, logarithm=True, reduce=True):
    condition = tf.shape(value).__len__() < tf.shape(prob).__len__()

    if condition:
        value = value[:, tf.newaxis, ...]
        reduce_start = 2
    else:
        reduce_start = 1

    rslt = value * tf.math.log(prob + OVERFLOW_MARGIN) + (-value + 1.) * tf.math.log(-prob + 1. + OVERFLOW_MARGIN)
    # noinspection DuplicatedCode
    rslt = tf.reduce_sum(rslt, axis=list(range(reduce_start, len(rslt.shape))))

    if condition:
        rslt = tf.reduce_mean(rslt, axis=-1)

    if not logarithm:
        rslt = tf.exp(rslt)

    return tf.reduce_mean(rslt) if reduce else rslt
