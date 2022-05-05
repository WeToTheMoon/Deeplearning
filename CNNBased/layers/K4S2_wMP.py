"""
About this layer Stack:
HIGH VARIENCE

Range: 0.87-0.93
Variance: ~0.2
Max: 0.932
"""

from keras import losses, layers

layers = [
    layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', input_shape=(256,256,3)),
    layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu'),
    layers.Conv2D(filters=256, kernel_size=4, strides=2, activation='relu'),
    layers.Conv2D(filters=512, kernel_size=4, strides=2, activation='relu'),
    layers.MaxPool2D(pool_size=2),

    layers.Dropout(rate=0.25),
    layers.Flatten(),

    layers.Dense(2048, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(3, activation='softmax'),
]