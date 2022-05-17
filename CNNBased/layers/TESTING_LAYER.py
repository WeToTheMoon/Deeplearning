"""

"""


from keras import layers, losses

layers = [
    layers.Conv2D(filters=64, kernel_size=10, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=8, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=6, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=4, strides=3, padding='same', activation='relu'),
    layers.BatchNormalization(),

    layers.Flatten(),

    # layers.Dense(2048, activation='relu'),
    # layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),

    layers.Dense(3, activation='softmax'),
]