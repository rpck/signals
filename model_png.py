import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model


input_shape = (2, 512, 1)

model = Sequential()
model.add(Conv2D(16, (1, 1), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(5, activation='softmax'))

model.load_weights('model.h5')

# print model
plot_model(model, to_file="model.png", show_shapes=True)

print("Model saved to png file.")