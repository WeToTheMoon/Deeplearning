import keras_tuner as kt
import keras
import tensorflow as tf
import numpy as np

from ImageManager import ImageManager

from keras import layers, losses



imageManager = ImageManager(r'C:\Users\achan\Documents\Coding\Python\ZachML\Dataset')
xTRAIN, xVAL, yTRAIN, yVAL = imageManager.getImages()

def build_model(hp):
    model = keras.Sequential(
        layers= [
            layers.Conv2D(filters=64, kernel_size=10, strides=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=8, strides=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=6, strides=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=4, strides=3, padding='same', activation='relu'),
            layers.BatchNormalization(),

            layers.Flatten(),

            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),

            layers.Dense(3, activation='softmax'),
        ]
    )

    # model.add(keras.layers.Dense(
    #     hp.Choice('units', [8, 16, 32]),
    #     activation='relu'))
    
    model.compile(
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321140/
        # Adam: 0.78-0.83
        # RMSprop: 0.73-0.81
        # Adamax:  0.74-0.78
        # AdaGrad: N/A
        # NAG: N/A
        # Gradient Decent: N/A
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

tuner.search(xTRAIN, yTRAIN, epochs=50, validation_data=(xVAL, yVAL))
best_model = tuner.get_best_models()[0]

print(best_model)

