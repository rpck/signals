import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
import json

# Open the model from a .json file
with open('models\mods\VT_QAM16_QAM64.json') as json_data:
    model_json = json.load(json_data)
model = model_from_json(json.dumps(model_json))

# Load the model weights from a .h5 file
model.load_weights('models\mods\VT_QAM16_QAM64.h5')

# -----------------------------------------------------

# Get the layer weights as a numpy array
weights = model.get_weights()

print(weights[0].shape)
print(weights[1].shape)
print(weights[2].shape)
print(weights[3].shape)
print(weights[4].shape)
print(weights[5].shape)
print(weights[6].shape)

# Create the figure to display on the plot
fig = plt.figure()

# Reshape weights into 2D array and add the heatmap of the first layer's weights
ax1 = fig.add_subplot(711)
layer1 = weights[0].reshape(1L, 8L)
ax1.imshow(layer1, cmap='hot', interpolation='nearest')

# Second layer
layer2 = weights[1].reshape(1L, 8L)
ax2 = fig.add_subplot(712)
ax2.imshow(layer2, cmap='hot', interpolation='nearest')

# Third layer, etc.
layer3 = weights[2].reshape(2L, 32L)
ax3 = fig.add_subplot(713)
ax3.imshow(layer3, cmap='hot', interpolation='nearest')

layer4 = weights[3].reshape(1L, 8L)
ax4 = fig.add_subplot(714)
ax4.imshow(layer4, cmap='hot', interpolation='nearest')

layer5 = weights[4].reshape(256L, 4000L)
ax5 = fig.add_subplot(715)
ax5.imshow(layer5, cmap='hot', interpolation='nearest')

layer6 = weights[5].reshape(2L, 64L)
ax6 = fig.add_subplot(716)
ax6.imshow(layer6, cmap='hot', interpolation='nearest')

layer7 = weights[6].reshape(5L, 128L)
ax7 = fig.add_subplot(717)
ax7.imshow(layer7, cmap='hot', interpolation='nearest')


# Show all 7 subplots
plt.show()

