"""
About this layer Stack:
low variance

Range: 0.87-0.89
Variance: ~0.1
Max: 0.893
"""

from keras import losses, layers

layers = [
    layers.Conv2D(filters=64, kernel_size=4, strides=4, padding='same', activation='relu', input_shape=(256,256,3)),
    layers.Conv2D(filters=128, kernel_size=4, strides=4, padding='same', activation='relu'),
    layers.Conv2D(filters=256, kernel_size=4, strides=4, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernel_size=4, strides=4, padding='same', activation='relu'),

    layers.Flatten(),

    layers.Dense(2048, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')
]