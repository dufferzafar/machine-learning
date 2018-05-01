from keras.models import Sequential

from keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization, Dropout,
    Flatten, Dense, Activation
)

INPUT_SHAPE = (28, 28, 1)


def keras_basic_cnn():

    net = Sequential()

    net.add(Conv2D(32, kernel_size=(5, 5),
                   activation='relu',
                   input_shape=INPUT_SHAPE))

    net.add(MaxPooling2D(pool_size=(2, 2)))

    net.add(Flatten())

    net.add(Dense(256, activation='relu'))
    net.add(Dense(20, activation='softmax'))

    return net


def keras_mnist_cnn():

    net = Sequential()

    net.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                   input_shape=INPUT_SHAPE))
    net.add(Conv2D(64, (3, 3), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    net.add(Dropout(0.25))

    net.add(Flatten())

    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.5))

    net.add(Dense(20, activation='softmax'))

    return net


def keras_alexnet():

    net = Sequential()

    net.add(Conv2D(64, (8, 8), activation='relu',
                   input_shape=INPUT_SHAPE))

    net.add(Conv2D(192, (5, 5), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))

    net.add(Conv2D(384, (2, 2), activation='relu'))
    net.add(Conv2D(256, (2, 2), activation='relu'))
    net.add(Conv2D(512, (2, 2), activation='relu'))

    net.add(Flatten())

    net.add(Dense(4096, activation='relu'))
    net.add(Dropout(0.5))

    net.add(Dense(4096, activation='relu'))
    net.add(Dropout(0.5))

    net.add(Dense(20, activation='softmax'))

    return net


def keras_vgg_13():

    net = Sequential()

    # Block 1

    net.add(Conv2D(64, (3, 3), padding='same', input_shape=INPUT_SHAPE))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Conv2D(64, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    net.add(Dropout(0.2))

    # Block 2

    net.add(Conv2D(128, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Conv2D(128, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    net.add(Dropout(0.2))

    # Block 3
    net.add(Conv2D(256, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Conv2D(256, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    net.add(Dropout(0.2))

    # Block 4
    net.add(Conv2D(512, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Conv2D(512, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    net.add(Dropout(0.2))

    # Block 5
    net.add(Conv2D(512, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    net.add(Conv2D(512, (3, 3), padding='same'))
    net.add(BatchNormalization())
    net.add(Activation('relu'))

    # Disable this pooling layer
    # net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # net.add(Dropout(0.2))

    net.add(Flatten())

    # net.add(Dropout(0.2))
    # net.add(Dense(512, activation='relu'))

    # net.add(Dropout(0.2))
    # net.add(Dense(512, activation='relu'))

    net.add(Dense(20, activation='softmax'))

    return net


# Large number of parameters (no BatchNormalization)
# Also called keras_vgg_stopped
def keras_vgg_19_small():

    net = Sequential()

    # Block 1
    net.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                   input_shape=INPUT_SHAPE))
    net.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    net.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    net.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    net.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    net.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    net.add(Dropout(0.3))

    # Block 4
    net.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    net.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    net.add(MaxPooling2D((2, 2), strides=(2, 2)))
    net.add(Dropout(0.3))

    net.add(Flatten())

    net.add(Dropout(0.3))
    net.add(Dense(4096, activation='relu'))

    net.add(Dropout(0.3))
    net.add(Dense(4096, activation='relu'))

    net.add(Dense(20, activation='softmax'))

    return net


if __name__ == '__main__':

    from keras.utils import plot_model

    archs = [
        # keras_basic_cnn, keras_mnist_cnn,
        keras_alexnet,
        keras_vgg_13, keras_vgg_19_small
    ]

    for arch in archs:
        name = arch.__name__

        print("##################")
        print("")
        print(name)

        net = arch()

        plot_model(net, to_file=name + ".png", show_shapes=True, show_layer_names=False)
        # net.summary()

        # net.save("output/" + name + ".h5")
