import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
import mat73
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models, losses

labels = ['glioma', 'meningioma', 'normal', 'pituitary']
x_train = []
y_train = []
x_test = []
y_test = []
i_size = 224

# for i in labels:
#     folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\dataset2\trainm', i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = mat73.loadmat(os.path.join(folderPath,j))
#         img = np.zeros((i_size,i_size,3))
#         x_train.append(img)
#         y_train.append(i)

for i in labels:
    folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\dataset3\Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(i_size,i_size))
        x_train.append(img)
        y_train.append(i)

# for i in labels:
#     folderPath = os.path.join(r'C:\Users\kesch\OneDrive\Desktop\dataset2\testm', i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = mat73.loadmat(os.path.join(folderPath,j))
#         img = np.zeros((i_size,i_size,3))
#         x_test.append(img)
#         y_test.append(i)

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

X_train = np.array(x_train) / 255.
X_test = np.array(x_test) / 255.
datagen = ImageDataGenerator(rotation_range=10,
shear_range=0.1,
zoom_range=0.1,
width_shift_range=0.15,
height_shift_range=0.15,
horizontal_flip=True)

datagen.fit(x_train)

model = models.Sequential()
model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(x_train,y_train,batch_size=64),validation_data=(x_test,y_test),epochs=100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()