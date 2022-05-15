import os
from warnings import filterwarnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, losses
from sklearn.metrics import classification_report,confusion_matrix
import h5py
from sklearn.model_selection import train_test_split
import seaborn as sns

labels = ['glioma', 'meningioma', 'pituitary']
x_train = []
y_train = []
i_size = 256

for i in labels:
    folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Documents\MATLAB\tumorpng2', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(i_size,i_size))
        x_train.append(img)
        y_train.append(i)
      
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train, test_size=0.2,random_state=14)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

x_train = np.array(x_train) / 255.
x_test = np.array(x_test) / 255.
datagen = ImageDataGenerator(rotation_range=15,
shear_range=0.1,
zoom_range=0.1,
width_shift_range=0.15,
height_shift_range=0.15,
horizontal_flip=True)

datagen.fit(x_train)


model = models.Sequential()
model.add(layers.Conv2D(64, 10, strides=3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, 8, strides=3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, 6, strides=3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(512, 4, strides=3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train,y_train, validation_data=(x_test,y_test), verbose = 1, epochs=75, batch_size=32)

model.summary()