import os
from datetime import datetime

import cv2
import keras
import keras.optimizers as optimizers # Dont delete me
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from CNNBased.ImageManager import ImageManager
from keras import losses, layers


class CNN:
    def __init__(self) -> None:
        self.layers = [
            layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', input_shape=(256,256,3)),
            layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu'),
            # layers.MaxPool2D(pool_size=2),
            layers.Conv2D(filters=256, kernel_size=4, strides=2, activation='relu'),
            # layers.MaxPool2D(pool_size=2),
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

    def start(self, filename:str = rf'results\cnn\Results {datetime.now().strftime(r"%Y_%m_%d-%I%M%S_%p")}.xlsx', path: str = r'C:\Users\kesch\OneDrive\Documents\MATLAB\tumorpng1', saveOutput: bool = True) -> None:
        """
        Starts the compiling and running of the CNN model

        ### Paramaters
        ---
        @path: Path to tumor images -> defaults to `C:/Users/kesch/OneDrive/Documents/MATLAB/tumorpng1`
        @save: bool: save results to excel file or not
        @filename: what to name excel sheet. Defaults to time of compliation

        """
        print(
            f"Starting CNN with arguments: \n"
            f"Save Output: {saveOutput}"
        )

        imageManager = ImageManager(path)
        xTRAIN, xVAL, yTRAIN, yVAL = imageManager.images()


        model = keras.Sequential(name="Brain-Cancer-Detector-Thing", layers=self.layers)

        model.compile(
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321140/
            # Adam: 0.78-0.83
            # RMSprop: 0.73-0.81
            # Adamax:  0.74-0.78
            # AdaGrad: N/A
            # NAG: N/A
            # Gradient Decent: N/A
            optimizer='Adam',
            loss=losses.categorical_crossentropy,
            metrics=['accuracy']
        )

        results = model.fit(
            xTRAIN,
            yTRAIN,
            validation_data=(xVAL,yVAL),
            verbose = 1,
            epochs=50,
            batch_size=32
        )

        if saveOutput:
            with open(filename, mode='wb') as file:
                print(f"Save Staus: {saveOutput}, Saving Results to {filename}")
                results = pd.DataFrame(results.history)
                results.to_excel(file)
                file.close()
        else:
            print(f"Save Staus: {saveOutput}, skipping")