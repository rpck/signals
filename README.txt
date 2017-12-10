# Signals CNN

To run the network, change any variables in `main.py` and run. The script will save both
the network model (the actual structure of the neurons/layers) in a JSON file, and
its corresponding weights in a H5 file.

To visualize, run the `visualize.py` script.

The visualization script reads from a very small subset of our full dataset. This can be
located in the `small_dataset` directory.

The `data_processing` directory holds MATLAB scripts that actually processed data from GNURadio.

The `models` directory holds a bunch of pre-trained networks on the main dataset. The `Conv8`
model was trained on the entire dataset with a kernel size of `(1, 1)` in the convolutional
layers. The `BigKernel8` model was also trained on the dataset, but the first Conv2D layer
had a kernel size of (2, 2). A README can be found inside the `models/mod` directory further
detailing the models.
