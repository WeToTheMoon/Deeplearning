"""
About this layer Stack:
High number of epochs for training

Range: 0.90-0.94.5
Variance: ~0.1
Max: 95.4
"""


from keras import layers, losses

layers = [
    layers.Conv2D(filters=64, kernel_size=10, strides=3, padding='same', activation='relu', input_shape=(256,256,3)),
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=8, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=6, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=4, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(3, activation='softmax'),
]