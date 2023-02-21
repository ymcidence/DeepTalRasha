import tensorflow as tf
from talrasha.functional import gaussian_kld


def test_kl():
    m1 = tf.constant([[2., 2., 2.], [1, 1, 1]], tf.float32)
    v1 = tf.ones_like(m1, tf.float32) * -.5

    m2 = tf.zeros_like(m1, tf.float32)
    v2 = tf.zeros_like(v1, tf.float32)

    d1 = gaussian_kld(m1, v1)
    d2 = gaussian_kld(m1, v1, m2, v2)

    print(d1)
    print(d2)
    return tf.abs(d1 - d2).numpy() < 1e-6


if __name__ == '__main__':
    test_kl()
