def get_dataset(num_classes, rank=0, size=1):

  from tensorflow import keras

  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)

  x_train = x_train[rank::size]

  y_train = y_train[rank::size]

  x_test = x_test[rank::size]

  y_test = y_test[rank::size]

  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

  x_train = x_train.astype('float32')

  x_test = x_test.astype('float32')

  x_train /= 255

  x_test /= 255

  y_train = keras.utils.to_categorical(y_train, num_classes)

  y_test = keras.utils.to_categorical(y_test, num_classes)

  return (x_train, y_train), (x_test, y_test)



def get_model(num_classes):

  from tensorflow.keras import models

  from tensorflow.keras import layers

  

  model = models.Sequential()

  model.add(layers.Conv2D(32, kernel_size=(3, 3),

                   activation='relu',

                   input_shape=(28, 28, 1)))

  model.add(layers.Conv2D(64, (3, 3), activation='relu'))

  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Dropout(0.25))

  model.add(layers.Flatten())

  model.add(layers.Dense(128, activation='relu'))

  model.add(layers.Dropout(0.5))

  model.add(layers.Dense(num_classes, activation='softmax'))

  return model


# Specify training parameters

batch_size = 128

epochs = 1

num_classes = 10

 

def train(learning_rate=1.0):

  from tensorflow import keras

  

  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)

  model = get_model(num_classes)

 

  # Specify the optimizer (Adadelta in this example), using the learning rate input parameter of the function so that Horovod can adjust the learning rate during training

  optimizer = keras.optimizers.Adadelta(lr=learning_rate)

 

  model.compile(optimizer=optimizer,

                loss='categorical_crossentropy',

                metrics=['accuracy'])

 
  print(x_train.shape)
  print(type(x_train))
  print(y_train.shape)
  print(type(y_train))

  
  model.fit(x_train, y_train,

            batch_size=batch_size,

            epochs=epochs,

            verbose=2,

            validation_data=(x_test, y_test))

  return model



model = train(learning_rate=0.1)




