import numpy as np

import cv2

from sklearn.model_selection import train_test_split

from common import load_data, write_csv, normalize

import keras
from k_arch import *


trX, trY, tsX = load_data()
classes, trYi = np.unique(trY, return_inverse=True)


def keras_load_data_split(trX, trYi, tsX, clean=True):

    # Image Cleaning via Erosion
    if clean:
        kernel = np.ones((2, 1))
        trX = np.array([cv2.erode(img, kernel)
                        for img in trX]).reshape(trX.shape)
        tsX = np.array([cv2.erode(img, kernel)
                        for img in tsX]).reshape(tsX.shape)

    # zero mean, unit variance
    trX = normalize(trX)
    tsX = normalize(tsX)

    # Split datasets
    x_test = tsX
    x_train, x_val, y_train, y_val = train_test_split(
        trX, trYi, test_size=0.3
    )

    # Reshape
    input_shape = (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_val = x_val.reshape(x_val.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)

    print('x_train shape:', x_train.shape)

    print(x_train.shape[0], 'training samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 20)
    y_val = keras.utils.to_categorical(y_val, 20)

    return x_train, y_train, x_val, y_val, x_test


def keras_load_data(trX, trYi, tsX, clean=True):

    # Image Cleaning via Erosion
    if clean:
        kernel = np.ones((2, 1))
        trX = np.array([cv2.erode(img, kernel)
                        for img in trX]).reshape(trX.shape)
        tsX = np.array([cv2.erode(img, kernel)
                        for img in tsX]).reshape(tsX.shape)

    # zero mean, unit variance
    trX = normalize(trX)
    tsX = normalize(tsX)

    # Reshape
    input_shape = (28, 28, 1)
    trX = trX.reshape(trX.shape[0], *input_shape)
    tsX = tsX.reshape(tsX.shape[0], *input_shape)
    trYc = keras.utils.to_categorical(trYi, 20)

    print(trX.shape[0], 'training samples')
    print(tsX.shape[0], 'test samples')

    return trX, trYc, tsX


def train_keras_cnn(arch_name="keras_alexnet"):

    batch_size = 128
    epochs = 10

    x_train, y_train, x_val, y_val, x_test = keras_load_data_split(trX, trYi, tsX)
    # x_train, y_train, x_test = keras_load_data(trX, trYi, tsX)

    arch = globals()[arch_name]

    net = arch()

    net.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        # optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy']
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.001,
            patience=4, verbose=1, mode='auto'
        )
    ]

    net.fit(
        x_train, y_train, verbose=2,
        batch_size=batch_size,

        initial_epoch=0, epochs=epochs,

        callbacks=callbacks_list,
        validation_data=(x_val, y_val)
    )

    tsP = classes[net.predict_classes(x_test)]
    write_csv("keras_vgg_13_cnn.csv", tsP)


def load_model(model_file):
    pass


def save_model(net):
    pass


if __name__ == '__main__':
    train_keras_cnn()
