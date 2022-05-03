import os

import cv2
import keras
import keras.utils.np_utils as np_utils
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras import layers, losses, models
from tqdm import tqdm

labels = ['glioma', 'meningioma', 'pituitary']
y_train, x_train,  = [], []
y_train_new, y_test_new = [], []


for i in labels:
    folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Documents\MATLAB\tumorpng1', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img,(256,256))
        x_train.append(img)
        y_train.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train, test_size=0.2,random_state=14)

for i in y_train:
    y_train_new.append(labels.index(i))

y_train = y_train_new
y_train = np_utils.to_categorical(y_train)

for i in y_test:
    y_test_new.append(labels.index(i))

y_test = y_test_new
y_test = np_utils.to_categorical(y_test)

x_train = np.array(x_train) / 255.
x_test = np.array(x_test) / 255.

datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True
)

datagen.fit(x_train)

model = keras.Sequential()
model.add(layers.Conv2D(96, 4, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(96, 4, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(4, strides=2))
model.add(layers.Conv2D(96, 4, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(96, 4, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(96, 4, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(96, 4, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train,y_train, validation_data=(x_test,y_test), verbose = 1, epochs=50, batch_size=32)

model.summary()