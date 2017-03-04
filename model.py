import itertools
from keras.layers import (Activation, Convolution1D, Dense, Dropout, Flatten,
                          MaxPooling1D)
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np


def prep_y(y):
    return np.array(y), 1
    # # # # # # # # # multiple classes
    # y, num_classes = prep_y(y)
    # if num_classes != 2:
    #     raise Exception("Not supporting more than 2 classes right now")
    # if isinstance(y, list):
    #     num_classes = len(set(y))
    # elif len(y.shape) == 1:
    #     num_classes = 2
    # elif len(y.shape) == 2 and y.shape[1] == 1:
    #     num_classes = len(set(y))
    # else:
    #     num_classes = y.shape[1]
    # y = np_utils.to_categorical(y, num_classes)
    # return y, num_classes


def get_binary_classification_model(X, y, layers=None, dense=512, dense_dropout=0.1):
    model = Sequential()

    y, num_classes = prep_y(y)

    if layers is None:
        layers = [conv1(64, 3, 0), conv1(64, 3, 0.1)]

    for num, layer in enumerate(layers):
        dropout = layer.pop("dropout")
        layer["activation"] = "relu"
        # only 1 layer
        if layer["nb_filter"] == 0:
            continue
        if num == 0:
            layer["input_shape"] = X.shape[1:]
        model.add(Convolution1D(**layer))
        model.add(MaxPooling1D(pool_length=2))
        if dropout != 0:
            model.add(Dropout(dropout))

    # fully connected
    model.add(Flatten())
    model.add(Dense(dense, activation="relu"))
    model.add(Dropout(dense_dropout))

    # to prediction
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy',
                  optimizer="rmsprop",
                  metrics=['accuracy'])
    return model, y


def get_multi_classification_model(X, y, layers=None, dense=512, dense_dropout=0.1):
    model = Sequential()

    y, num_classes = np_utils.to_categorical(np.arange(len(y)), len(y)), len(y)

    if layers is None:
        layers = [conv1(64, 3, 0), conv1(64, 3, 0.1)]

    for num, layer in enumerate(layers):
        dropout = layer.pop("dropout")
        layer["activation"] = "relu"
        # only 1 layer
        if layer["nb_filter"] == 0:
            continue
        if num == 0:
            layer["input_shape"] = X.shape[1:]
        model.add(Convolution1D(**layer))
        model.add(MaxPooling1D(pool_length=2))
        if dropout != 0:
            model.add(Dropout(dropout))

    # fully connected
    model.add(Flatten())
    model.add(Dense(dense, activation="relu"))
    model.add(Dropout(dense_dropout))

    # to prediction
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer="rmsprop",
                  metrics=['accuracy'])
    return model, y


def conv1(nb_filter, filter_length, dropout):
    return {"nb_filter": nb_filter, "filter_length": filter_length, "dropout": dropout}


def grid(conv=conv1):
    nb_filter1 = [16, 32, 64, 128]
    filter_length1 = [2, 3, 4]
    dropout1 = [0, 0.05, 0.1, 0.2]

    nb_filter2 = [0, 16, 32]
    filter_length2 = [2, 3, 4]
    dropout2 = [0, 0.05, 0.1, 0.2]

    product = itertools.product(nb_filter1, filter_length1, dropout1,
                                nb_filter2, filter_length2, dropout2)
    for n1, f1, d1, n2, f2, d2 in product:
        layers = [conv(n1, f1, d1), conv(n2, f2, d2)]
        yield layers
