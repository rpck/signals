import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import json

with open('models\model.json') as json_data:
    model_json = json.load(json_data)

model = model_from_json(json.dumps(model_json))

model.load_weights('models\model.h5')

# -----------------------------------------------------

weights = model.get_weights()


fig = plt.figure()

ax1 = fig.add_subplot(711)
layer1 = weights[0].reshape(1L, 16L)
print(layer1)
ax1.imshow(layer1, cmap='hot', interpolation='nearest')

layer2 = weights[1].reshape(1L, 16L)
ax2 = fig.add_subplot(712)
ax2.imshow(layer2, cmap='hot', interpolation='nearest')

layer3 = weights[2].reshape(4L, 64L)
ax3 = fig.add_subplot(713)
ax3.imshow(layer3, cmap='hot', interpolation='nearest')

layer4 = weights[3].reshape(1L, 16L)
ax4 = fig.add_subplot(714)
ax4.imshow(layer4, cmap='hot', interpolation='nearest')

layer5 = weights[4].reshape(128L, 4000L)
ax5 = fig.add_subplot(715)
ax5.imshow(layer5, cmap='hot', interpolation='nearest')

layer6 = weights[5].reshape(2L, 64L)
ax6 = fig.add_subplot(716)
ax6.imshow(layer6, cmap='hot', interpolation='nearest')

layer7 = weights[6].reshape(5L, 128L)
ax7 = fig.add_subplot(717)
ax7.imshow(layer7, cmap='hot', interpolation='nearest')


plt.show()

