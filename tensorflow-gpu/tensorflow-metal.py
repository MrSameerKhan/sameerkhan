import tensorflow as tf
import numpy as np

# Print TensorFlow version and available devices
print("TensorFlow Version:", tf.__version__)
print("Available Devices:", tf.config.list_physical_devices())

# Create dataset
X = np.arange(1, 101, step=0.1).reshape(-1, 1)  # Reshaped to (num_samples, 1)
y = X + 10  # Vectorized operation

# Convert to TensorFlow tensors
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=['mae']  # Using 'mae' instead of 'mean_absolute_error'
)

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Make predictions (fixed shape issue)
predictions = model.predict(np.array([[10], [20], [30]]), verbose=0)  
print("Predictions:", predictions.flatten())


print("END")