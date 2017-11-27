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
    training_set = []
    training_labels = []

    validation_set = []
    validation_labels = []

    testing_set = []
    testing_labels = []
    i = 0
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
            if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
                training_set.append(data)
                training_labels.append(label_int)
            elif i % 5 == 3:
                validation_set.append(data)
                validation_labels.append(label_int)
            elif i % 5 == 4:
                testing_set.append(data)
                testing_labels.append(label_int)
            i += 1

    training_set = np.array(training_set)

    validation_set = np.array(validation_set)
    testing_set = np.array(testing_set)

    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)
    testing_labels = np.array(testing_labels)

    # Shuffle the dataset and labels with the same permutation
    train_perm = np.random.permutation(training_set.shape[0])
    valid_perm = np.random.permutation(validation_set.shape[0])
    test_perm = np.random.permutation(testing_set.shape[0])

    new_train_set = training_set[train_perm]
    new_train_labels = training_labels[train_perm]

    new_valid_set = validation_set[valid_perm]
    new_valid_labels = validation_labels[valid_perm]

    new_test_set = testing_set[test_perm]
    new_test_labels = testing_labels[test_perm]

    return (new_train_set, new_train_labels), (new_valid_set, new_valid_labels), (new_test_set, new_test_labels)

loaded_train_set, loaded_valid_set, loaded_test_set = load_dataset('test_dataset')
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
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


