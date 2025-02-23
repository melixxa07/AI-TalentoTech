import tensorflow as tf
import numpy as np

print('TensorFlow version:', tf.__version__)
print('Numpy version:', np.__version__)

# Create a simple tensor
tensor = tf.constant([[1, 2], [3, 4]])
print('Tensor:', tensor)

# Do a simple operation
result = tf.matmul(tensor, tf.transpose(tensor))
print('Result of matrix multiplication:', result)