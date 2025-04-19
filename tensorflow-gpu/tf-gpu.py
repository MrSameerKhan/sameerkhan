import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))


import tensorflow as tf

# Check available devices
print("Available devices:", tf.config.list_physical_devices())

# Check if GPU is being used
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("TensorFlow is using GPU:", gpu_devices)
    print("GPU Details:", tf.config.experimental.get_device_details(gpu_devices[0]))
else:
    print("No GPU detected.")

tf.keras.mixed_precision.set_global_policy('mixed_float16')


import tensorflow as tf
import time

size = 10000  # Increase matrix size
a = tf.random.uniform([size, size], dtype=tf.float16)
b = tf.random.uniform([size, size], dtype=tf.float16)

# CPU Computation
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    end = time.time()
    print("CPU Time:", end - start)

# GPU Computation
with tf.device('/GPU:0'):
    start = time.time()
    d = tf.matmul(a, b)
    end = time.time()
    print("GPU Time:", end - start)


