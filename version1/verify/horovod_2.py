from tensorflow import keras

rank = 0 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)

print(x_train.shape)
print(type(x_train))

print(x_train)