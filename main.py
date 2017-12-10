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

# Some constants to define our data
rows = 2
pts_per_sample = 500
num_categories = 5
batch_size = 64
epochs = 5
model_name = 'model_name'

# Labels for our classifications, looking back we could have just
# used a dictionary.
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
    # We load everything into one big "main set," shuffle this
    # set, and then allocate 60% to a training set, 20% to a
    # validation set, and 20% to a test set.
    main_set = []
    main_labels = []
    # Searches the specified dataset directory
    for f in os.listdir(dataset_dir):
        # Our signals have a real and imaginary component
        real = None
        imag = None
        label = f[:-4]
        print('Loading', label)
        infile = open(os.path.join(dataset_dir, f))

        # This is the piece of code that defined what data the
        # network was actually trained on.
        # Uncomment to only load QAM64 and QPSK
        #if label != 'QAM64' and label != 'QPSK':
        #   continue

        # Iterate through a data file. The setup assumes one signal
        # is stored in two lines- the first line holds the real data,
        # and the second line holds imaginary data. The lines are 500
        # CSV's. One input file may have many such pairs.
        for line in infile:
            real = np.fromstring(line, dtype=np.float64, sep=',')
            imag = np.fromstring(next(infile), dtype=np.float64, sep=',')
            data = np.stack((real, imag))
            label_int = label_to_int(label)
            # Load the data and append to the main set
            main_set.append(data)
            main_labels.append(label_int)

    # Uses numpy to shuffle the set and labels. We create a permutation
    # So that the corresponding labels for each signal sample is preserved.
    main_set = np.array(main_set)
    main_labels = np.array(main_labels)
    main_perm = np.random.permutation(main_set.shape[0])

    # The new shuffled set after applying the SAME PERMUTATION on both the
    # data and its labels.
    new_main_set = main_set[main_perm]
    new_main_labels = main_labels[main_perm]

    size = len(new_main_set)
    training = int(0.60 * size)
    validating = int(0.20 * size)
    testing = int(0.20 * size)

    # Split into different sets.
    training_set = new_main_set[:training]
    validation_set = new_main_set[training:training + validating]
    testing_set = new_main_set[training + validating:training + validating + testing]

    training_labels = new_main_labels[:training]
    validation_labels = new_main_labels[training:training + validating]
    testing_labels = new_main_labels[training + validating:training + validating + testing]

    # Return as a tuple
    return (training_set, training_labels), (validation_set, validation_labels), (testing_set, testing_labels)

# Loads the dataset and scrambles it into 3 separate sets
loaded_train_set, loaded_valid_set, loaded_test_set = load_dataset('new_dataset')
input_shape = (1, rows, pts_per_sample)

train_dataset = ()
valid_dataset = ()
test_dataset = ()

# 1 channel
if K.image_data_format() == 'channels_first':
    train_dataset = loaded_train_set[0].reshape(len(loaded_train_set[0]), 1, rows, pts_per_sample)
    valid_dataset = loaded_valid_set[0].reshape(len(loaded_valid_set[0]), 1, rows, pts_per_sample)
    test_dataset = loaded_test_set[0].reshape(len(loaded_test_set[0]), 1, rows, pts_per_sample)
    input_shape = (1, rows, pts_per_sample)
else:
    train_dataset = loaded_train_set[0].reshape(len(loaded_train_set[0]), rows, pts_per_sample, 1)
    valid_dataset = loaded_valid_set[0].reshape(len(loaded_valid_set[0]), rows, pts_per_sample, 1)
    test_dataset = loaded_test_set[0].reshape(len(loaded_test_set[0]), rows, pts_per_sample, 1)
    input_shape = (rows, pts_per_sample, 1)

train_labels = np_utils.to_categorical(loaded_train_set[1], num_categories)
valid_labels = np_utils.to_categorical(loaded_valid_set[1], num_categories)
test_labels = np_utils.to_categorical(loaded_test_set[1], num_categories)

# Define a Keras Sequential model
model = Sequential()
model.add(Conv2D(8, (1, 1), activation='relu', input_shape=input_shape))
model.add(Conv2D(8, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))

# Classifier stage
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(5, activation='softmax'))

# Report the structure of the network
model.summary()

# Create the model
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# Fit the data and validate/test
model.fit(train_dataset, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(valid_dataset, valid_labels))

score = model.evaluate(test_dataset, test_labels, batch_size=batch_size, verbose=0)

# Print out the performance
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Saves the model
# Serialize model to JSON
model_json = model.to_json()

with open("{0}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
    
# Serialize weights to HDF5
model.save_weights("{0}.h5".format(model_name))
print("Saved model to disk")
