import tensorflow as tf
import numpy as np
import os

# Generate some data
X_train = np.random.rand(1000, 1)
y_train = 2 * X_train + 1 + np.random.randn(1000, 1) * 0.1

# Create a simple neural network and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))  # input_shape is (1,) because we have one feature (X) and the comma (1,) is because input_shape must be a tuple
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Save the model
model_dir = '/models/model/1'
os.makedirs(model_dir, exist_ok=True)
model.save(model_dir)
print('Model saved to', model_dir)
# Note: The model is saved in the TensorFlow SavedModel format, which is a directory containing the model architecture and weights.