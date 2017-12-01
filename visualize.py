import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
import numpy as np
import argparse

def get_layer_outputs(model, test_data):
    outputs = [layer.output for layer in model.layers]
    functor = [K.function([model.input] + [K.learning_phase()], [out]) for out in outputs]
    layer_outs = [func([test_data, 1.]) for func in functor]
    return layer_outs

def plot_layer_outputs(model, layer_number, test_data):
    layer_outputs = get_layer_outputs(model, test_data)

    # What does this mean?
    print(layer_outputs[layer_number])

    for filter_num in range(0, 4):
        plt.figure(filter_num)
        img = layer_outputs[layer_number][0][0, filter_num,:,:]

        plt.plot(img[100:228,0], color = 'b')    ## real component of the raw IQ, only 128 samples
        plt.plot(img[100:228,1], color = 'g')    ## imag component of the raw IQ, only 128 samples

        plt.plot(img[:,0], color = 'b')    ## real component of the raw IQ
        plt.plot(img[:,1], color = 'g')    ## imag component of the raw IQ

        # fft of the raw IQ
        data_complex = img[:,0] + 1j*img[:,1]
        img_fft = np.fft.fft(data_complex)
        img_rearrange = np.hstack((np.abs(img_fft)[512:],np.abs(img_fft)[0:512]))
        plt.plot(img_rearrange)
        plt.ylim(ymax = 100)

        ## display ffted data
        data_complex = img[:,0] + 1j*img[:,1]
        img_rearrange = np.hstack((np.abs(data_complex)[512:],np.abs(data_complex)[0:512]))
        plt.plot(img_rearrange)
        plt.ylim(ymax =20)

    plt.show()
