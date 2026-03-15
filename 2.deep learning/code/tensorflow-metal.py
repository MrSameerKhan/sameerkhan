import tensorflow as tf
print(tf.__version__)

print(tf.config.list_physical_devices())

import numpy as np

X=np.arange(1,101,step=0.1)
y=[x+10 for x in X]

X=tf.cast(tf.constant(X), dtype=tf.float32)
y=tf.cast(tf.constant(y), dtype=tf.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(1,),activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mean_absolute_error,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=['mean_absolute_error']
)

model.fit(X,y,epochs=100)

model.predict([10,20,30])
print("END")