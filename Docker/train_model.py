import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(0) # This ensures that the random numbers generates are the same every time the code is run, making it easier to reproduce the results
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1)

# Create a simple neural network and complile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))  # input_shape is (1,) because we have one feature (X) and the comma (1,) is because input_shape must be a tuple
])

model.compile(optimizer='sgd', loss='mse')

# Train the model
history = model.fit(X, y, epochs=100, verbose=0)

# Plot the results
plt.scatter(X, y, label='Data')
plt.plot(X, model.predict(X), color='red', label='Predictions')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('/app/regression_plot.png')
print('Plot saved to /app/regression_plot.png')

# Print the model weights
print('Model weights:', model.get_weights())


# Note: When running a Docker container without a display server (like X11 
# on Linux or a GUI on Windows), Matplotlib cannot directly show plots using plt.show().
# Instead, you must save the plot as an image (e.g., plt.savefig()) and then view it outside the container.