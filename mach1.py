import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib as plt

number_epochs = 1
    
#Load dataset   
mnist = tf.keras.datasets.mnist 
(train_X, train_Y), (test_X, test_Y) = mnist.load_data() 
    
#Convert from int to float
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

#One hot encoding
train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)
    
#Normalise data
train_X = train_X / 255.0
test_X = test_X / 255.0

#Reshape dataset
train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))


model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile model
compile_model = model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
training = model.fit(train_X, train_Y, epochs = number_epochs)

val_loss, val_acc = model.evaluate(test_X, test_Y)
print(val_loss, val_acc)

model.save("epic_num_reader.model")
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([test_X])
print(predictions)

print(np.argmax(predictions[0]))

plt.imshow(test_X[0])
plt.show()
