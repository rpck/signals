import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
import json
import os

num_categories = 5

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

def load_dataset(dataset_dir):
    # Our classifications
    main_set = []
    main_labels = []

    for f in os.listdir(dataset_dir):
        real = None
        imag = None
        label = f[:-4]
        # % is the string formatting operator, like printf('Loading %s', label)
        print('Loading {0}...'.format(label))
        infile = open(os.path.join(dataset_dir, f))

        for line in infile:
            real = np.fromstring(line, dtype=float, sep=',')
            imag = np.fromstring(next(infile), dtype=float, sep=',')
            data = np.stack((real, imag))
            main_set.append(data)
            main_labels.append(label)

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

def get_layer_outputs(model, layer_number, test_data):
    outputs = [layer.output for layer in model.layers]
    functor = [K.function([model.input] + [K.learning_phase()], [out]) for out in outputs]
    layer_outs = [func([test_data, 1.]) for func in functor]
    return layer_outs

def plot_layer_outputs(model, layer_number, test_data, test_labels):
    # 1 corresponds to the first layer
    figures = 0
    layer_outputs = get_layer_outputs(model, 0, test_data)
    print(test_labels[0])
    for layer_out in layer_outputs:
        for filter_num in range(0, 16):
            figures += 1
            fig = plt.figure(figures)
            y = layer_out[0][0, filter_num,:,:]
            x = np.arange(y[0].shape[0])
            plt.plot(x, y[0], 'b')
            plt.plot(x, y[1], 'r')
            plt.ylim(ymax=0.4, ymin=0)
            fig.show()
        plt.show()

with open('models/model.json') as json_data:
    model_json = json.load(json_data)
model = model_from_json(json.dumps(model_json))
model.load_weights('models/model.h5')

loaded_train_set, loaded_valid_set, loaded_test_set = load_dataset('new_dataset')

train_dataset = ()
valid_dataset = ()
test_dataset = ()

# Even though we have channels_last in the config...
train_dataset = loaded_train_set[0].reshape(len(loaded_train_set[0]), 1, 2, 500)
valid_dataset = loaded_valid_set[0].reshape(len(loaded_valid_set[0]), 1, 2, 500)
test_dataset = loaded_test_set[0].reshape(len(loaded_test_set[0]), 1, 2, 500)

train_labels = loaded_train_set[1]
valid_labels = loaded_valid_set[1]
test_labels = loaded_test_set[1]

plot_layer_outputs(model, 0, test_dataset[0:100], test_labels)
