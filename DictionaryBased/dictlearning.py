import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tqdm import tqdm

from workers.ModelControl import ModelController

labels = ['glioma', 'meningioma', 'normal', 'pituitary']
x_train, y_train, x_test, y_test = [], [], [], []

imageSizePX = 224

for i in labels:
    trainingPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\test_dataset3\Training', i)
    for j in tqdm(os.listdir(trainingPath)):
        img = cv2.imread(os.path.join(trainingPath,j))
        img = cv2.resize(img,(imageSizePX,imageSizePX))
        x_train.append(img)
        y_train.append(i)
        
    testingPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\test_dataset3\Testing', i)
    for j in tqdm(os.listdir(testingPath)):
        img = cv2.imread(os.path.join(testingPath,j))
        img = cv2.resize(img,(imageSizePX,imageSizePX))
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

ModelController = ModelController()
ModelController.addModelLayers()
ModelController.compileModel()

dict_in = np.array(ModelController.predict(x_train))

print(dict_in)
# dictionary = dl.ksvd(dict_in, dictionary, 100, n_nonzero=8, n_threads=4, verbose=True)
# print(dictionary)