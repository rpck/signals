import numpy as np
import os
import keras
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import re

def load_dataset(dataset_dir):
    pass

# Use tanh instead of ReLU to prevent NaN errors
model = Sequential()
model.add(Conv2D(512,
        kernel_size=(6, 2),
        activation='tanh',
        padding='same',
        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=1,
        padding='same',
        data_format=None))
model.add(Dropout(0.25))
model.add(Conv2D(512,
        kernel_size=(2, 2),
        activation='tanh',
        padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=1,
        padding='same',
        data_format=None))
model.add(Flatten())

#"Squash" to probabilities
model.add(Dense(5, activation='softmax'))

model.summary()

# Use a Stochastic-Gradient-Descent as a learning optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Prevent kernel biases from being exactly 0 and giving nan errors
def constrainedCrossEntropy(ytrue, ypred):
    ypred = K.clip(ypred, 1e-7, 1e7)
    return losses.categorical_crossentropy(ytrue, ypred)

model.compile(loss=constrainedCrossEntropy,
              optimizer=sgd,
              metrics=['accuracy'])
