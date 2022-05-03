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
from tensorflow.python.keras import layers
from tqdm import tqdm


class CNN:
    def __init__(self) -> None:
        self.layers = [
            layers.Conv2D(64, 4, strides=4, padding='same', activation='relu'),
            layers.Conv2D(128, 4, strides=4, padding='same', activation='relu'),
            layers.Conv2D(256, 4, strides=4, padding='same', activation='relu'),
            layers.Conv2D(512, 4, strides=4, padding='same', activation='relu'),
            
            layers.Flatten(),
            
            layers.Dense(2048, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(3, activation='softmax')
        ]
        
    def start(self, path: str = r'C:\Users\kesch\OneDrive\Documents\MATLAB\tumorpng1', saveOutput: bool = True) -> None:
        """
        Starts the compiling and running of the CNN model

        ### Paramaters
        ---
        @path: Path to tumor images -> defaults to `C:/Users/kesch/OneDrive/Documents/MATLAB/tumorpng1`

        """
        print("Starting CNN")
        cancerTypes = ['glioma', 'meningioma', 'pituitary']

        typeList, imageList,  = [], []
        y_train_new, y_test_new = [], []

        for type in cancerTypes:
            print(f"Loading and pre-processing {type} images")
            folderPath = os.path.join(path, type)
            for j in tqdm(os.listdir(folderPath)):
                img = cv2.imread(os.path.join(folderPath, j))
                img = cv2.resize(img, (256,256))
                imageList.append(img)
                typeList.append(type)

        print("Caching Images")

        imageList = np.array(imageList)
        typeList = np.array(typeList)

        xTRAIN, xVAL, yTRAIN, yVAL = train_test_split(imageList, typeList, test_size=0.2, random_state=14)

        for i in yTRAIN:
            y_train_new.append(cancerTypes.index(i))

        yTRAIN = to_categorical(y_train_new)

        for i in yVAL:
            y_test_new.append(cancerTypes.index(i))

        yVAL = to_categorical(y_test_new)

        xTRAIN = xTRAIN / 255
        xVAL = xVAL / 255

        generatedData = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.1,
            zoom_range=0.1,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True
        )

        generatedData.fit(xTRAIN)
        # scale >=4
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
            loss='categorical_crossentropy',
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
            filename = f'results\cnn\Results {datetime.now().strftime(r"%Y_%m_%d-%I%M%S_%p")}.xlsx'
            with open(filename, mode='wb') as file:
                print(f"Save Staus: {saveOutput}, Saving Results to {filename}")
                results = pd.DataFrame(results.history)
                results.to_excel(file)
                file.close()
        else:
            print(f"Save Staus: {saveOutput}, skipping")
