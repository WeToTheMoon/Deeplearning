import os

import cv2 as cv2
import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from dictlearn import dictionary_learning
from sklearn.decomposition import DictionaryLearning
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tqdm import tqdm

labels = ['glioma', 'meningioma', 'normal', 'pituitary']
x_train = []
y_train = []
x_test = []
y_test = []
i_size = 224

for i in labels:
    folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\dataset3\Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(i_size,i_size))
        x_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\dataset3\Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(i_size,i_size))
        x_test.append(img)
        y_test.append(i)
      
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

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

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(96, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(96, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(4, strides=2))
model.add(layers.Conv2D(96, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(96, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(96, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(96, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

results = model.predict(x_train)

dictionary = dictionary_learning.dct_dict(64)