from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf

from layer.biased_transformer import BiasedTransformer

model = BiasedTransformer(128, 9, 8)
x = tf.ones([4, 8, 96])

y = model(x, x, training=True)
print(y)