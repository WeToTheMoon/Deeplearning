import sys

import tensorflow as tf
from numpy import ndarray
from tensorflow.python.keras import layers, models


class ModelController:
    def __init__(this):
        this.model = tf.keras.models.Sequential()

    def addModelLayers(this):
        this.model.add(layers.Conv2D(100, 3, strides=4, padding='same'))
        this.model.add(layers.Activation('relu'))
        this.model.add(layers.MaxPooling2D(3, strides=2))
        this.model.add(layers.Conv2D(100, 3, strides=4, padding='same'))
        this.model.add(layers.Activation('relu'))
        this.model.add(layers.MaxPooling2D(4, strides=2))
        this.model.add(layers.Conv2D(100, 3, strides=4, padding='same'))
        this.model.add(layers.Activation('relu'))
        this.model.add(layers.Conv2D(100, 3, strides=4, padding='same'))
        this.model.add(layers.Activation('relu'))
        this.model.add(layers.Conv2D(100, 3, strides=4, padding='same'))
        this.model.add(layers.Activation('relu'))
        this.model.add(layers.Conv2D(100, 3, strides=4, padding='same'))
        this.model.add(layers.Activation('relu'))
        this.model.add(layers.Flatten())

        this.layersAdded = True

    def compileModel(this):
        try:
            this.layersAdded
        except:
            sys.exit("No layers have been added to this model")

        this.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(this, set: ndarray):
        return this.model.predict(set)
