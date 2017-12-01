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

num_categories = 5
batch_size = 64
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
    return 5

#Training/Validation/Testing split, ~ 60/20/20
def load_dataset(dataset_dir):
    # Our classifications
    main_set = []
    main_labels = []

    for f in os.listdir(dataset_dir):
        real = None
        imag = None
        label = f[:-4]
        print(label)
        infile = open(os.path.join(dataset_dir, f))

        for line in infile:
            real = np.fromstring(line, dtype=float, sep=',')
            imag = np.fromstring(next(infile), dtype=float, sep=',')
            data = np.stack((real, imag))
            label_int = label_to_int(label)
            main_set.append(data)
            main_labels.append(label_int)

    main_set = np.array(main_set)
    main_labels = np.array(main_labels)
    main_perm = np.random.permutation(main_set.shape[0])

    new_main_set = main_set[main_perm]
    new_main_labels = main_labels[main_perm]

    size = len(new_main_set)
    training = int(0.60 * size)
    validating = int(0.20 * size)
    testing = int(0.20 * size)

    training_set = new_main_set[:training]
    validation_set = new_main_set[training:training + validating]
    testing_set = new_main_set[training + validating:training + validating + testing]

    training_labels = new_main_labels[:training]
    validation_labels = new_main_labels[training:training + validating]
    testing_labels = new_main_labels[training + validating:training + validating + testing]

    return (training_set, training_labels), (validation_set, validation_labels), (testing_set, testing_labels)

loaded_train_set, loaded_valid_set, loaded_test_set = load_dataset('new_dataset')
input_shape = (1, 2, 500)
train_dataset = ()
valid_dataset = ()
test_dataset = ()

if K.image_data_format() == 'channels_first':
    train_dataset = loaded_train_set[0].reshape(len(loaded_train_set[0]), 1, 2, 500)
    valid_dataset = loaded_valid_set[0].reshape(len(loaded_valid_set[0]), 1, 2, 500)
    test_dataset = loaded_test_set[0].reshape(len(loaded_test_set[0]), 1, 2, 500)
    input_shape = (1, 2, 500)
else:
    train_dataset = loaded_train_set[0].reshape(len(loaded_train_set[0]), 2, 500, 1)
    valid_dataset = loaded_valid_set[0].reshape(len(loaded_valid_set[0]), 2, 500, 1)
    test_dataset = loaded_test_set[0].reshape(len(loaded_test_set[0]), 2, 500, 1)
    input_shape = (2, 500, 1)

train_labels = np_utils.to_categorical(loaded_train_set[1], num_categories)
valid_labels = np_utils.to_categorical(loaded_valid_set[1], num_categories)
test_labels = np_utils.to_categorical(loaded_test_set[1], num_categories)

# Use tanh instead of ReLU to prevent NaN errors
model = Sequential()
model.add(Conv2D(16, (1, 1), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

model.fit(train_dataset, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(valid_dataset, valid_labels))

score = model.evaluate(test_dataset, test_labels, batch_size=batch_size, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Saves the model
#Serialize model to JSON
model_json = model.to_json()
with open("VT.json", "w") as json_file:
    json_file.write(model_json)
#Serialize weights to HDF5
model.save_weights("VT.h5")
print("Saved model to disk")
