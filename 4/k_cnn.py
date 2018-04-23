import numpy as np

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from common import load_data, write_csv


trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


def keras_load_data(trX, trYi, tsX):

    x_test = tsX
    x_train, x_val, y_train, y_val = train_test_split(
        trX, trYi, test_size=0.3
    )

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_val = x_val.reshape(x_val.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)

    print(x_train.shape[0], 'training samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 20)
    y_val = keras.utils.to_categorical(y_val, 20)

    return input_shape, x_train, y_train, x_val, y_val, x_test


def keras_mnist_cnn(input_shape):

    net = Sequential()

    net.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))

    net.add(Conv2D(64, (3, 3), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.25))

    net.add(Flatten())

    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.5))

    net.add(Dense(20, activation='softmax'))

    return net


def keras_deeper_cnn(input_shape):

    net = Sequential()

    net.add(Conv2D(64, (8, 8), activation='relu',
                   input_shape=input_shape))

    net.add(Conv2D(192, (5, 5), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.2))

    net.add(Conv2D(384, (2, 2), activation='relu'))
    net.add(Conv2D(256, (2, 2), activation='relu'))
    net.add(Conv2D(256, (2, 2), activation='relu'))

    net.add(Flatten())

    net.add(Dropout(0.3))
    net.add(Dense(2048, activation='relu'))

    net.add(Dropout(0.5))
    net.add(Dense(2048, activation='relu'))

    net.add(Dense(20, activation='softmax'))

    return net


def keras_basic_cnn(input_shape):

    net = Sequential()

    net.add(Conv2D(32, kernel_size=(5, 5),
                   activation='relu',
                   input_shape=input_shape))

    net.add(MaxPooling2D(pool_size=(4, 4)))

    net.add(Flatten())

    net.add(Dense(128, activation='relu'))
    net.add(Dense(20, activation='softmax'))

    return net


if __name__ == '__main__':

    batch_size = 128
    epochs = 25

    input_shape, x_train, y_train, x_val, y_val, x_test = keras_load_data(
        trX, trYi, tsX
    )

    # net = keras_deeper_cnn(input_shape)
    # net = keras_mnist_cnn(input_shape)
    net = keras_basic_cnn(input_shape)

    net.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.0001,
            patience=3, verbose=1, mode='auto'
        )
    ]

    net.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks_list,
        validation_data=(x_val, y_val)
    )

    tsP = classes[net.predict_classes(x_test)]
    write_csv("keras_deeper_cnn.csv", tsP)
