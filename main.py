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

batch_size = 32
epochs = 5

def label_to_int(label):
    if label == 'BPSK':
        return 0
    elif label == 'QAM16':
        return 1
    elif label == 'QAM64':
        return 2
    elif label == 'QPSK':
        return 3
    elif label == 'VT':
        return 4
    return 0

def load_dataset(dataset_dir):
    # Our classifications
    dataset = []
    labels = []
    test = []
    test_labels = []
    i = 0
    for f in os.listdir(dataset_dir):
        real = None
        imag = None
        label = f[:-4]
        infile = open(os.path.join(dataset_dir, f))
        for line in infile:
            real = np.fromstring(line, dtype=float, sep=',')
            imag = np.fromstring(next(infile), dtype=float, sep=',')
            data = np.stack((real, imag))
            label_int = label_to_int(label)
            if i % 8 == 0:
                test.append(data)
                test_labels.append(label_int)
            else:
                dataset.append(data)
                labels.append(label_int)
            i += 1
    return (np.array(dataset), np.array(labels)), (np.array(test), np.array(test_labels))

train_set, test_set = load_dataset('dataset')
train_dataset = train_set[0].reshape(len(train_set[0]), 1, 512, 2)
test_dataset = test_set[0].reshape(len(test_set[0]), 1, 512, 2)
input_shape = (1, 512, 2)
train_labels = np_utils.to_categorical(train_set[1], 5)
test_labels = np_utils.to_categorical(test_set[1], 5)

# Use tanh instead of ReLU to prevent NaN errors
model = Sequential()
model.add(Conv2D(32,
        kernel_size=(6, 2),
        activation='tanh',
        padding='same',
        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),
        strides=1,
        padding='same',
        data_format=None))
model.add(Dropout(0.25))
model.add(Conv2D(32,
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

model.fit(train_dataset, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(test_dataset, test_labels))
