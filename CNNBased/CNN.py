import os
from datetime import datetime

import cv2
import keras
import sys
import keras.optimizers as optimizers  # Dont delete me
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, losses
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from CNNBased.ImageManager import ImageManager
from CNNBased.ModelController import ModelController
from CNNBased.layers.layerLists import K4S2MP, K4S2MP_GRAY, K4S4, K4S4_GRAY


class CNN:
    def __init__(self) -> None:
        self.modelController = ModelController(K4S2MP_GRAY)
        self.model = self.modelController.getModel()

    def start(self, path: str = r'C:\Users\kesch\OneDrive\Documents\MATLAB\tumorpng1') -> None:
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
            f"Save Output: {saveOutput} \n"
            f"Save Filename: {filename} \n"
        )

        imageManager = ImageManager(path)
        xTRAIN, xVAL, yTRAIN, yVAL = imageManager.images(gray=True)

        self.model.compile(
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

        self.results = self.model.fit(
            xTRAIN,
            yTRAIN,
            validation_data=(xVAL,yVAL),
            verbose = 1,
            epochs= 50,
            batch_size= 32
        )

    def saveResults(self,  saveOutput: bool = False, filename:str = rf'results\cnn\Results {datetime.now().strftime(r"%Y_%m_%d-%I%M%S_%p")}.xlsx' ):
        try:
            self.results
        except:
            sys.exit("Must fit model before saving results")

        if saveOutput:
            with open(filename, mode='wb') as file:
                print(f"Save Staus: {saveOutput}, Saving Results to {filename}")
                self.results = pd.DataFrame(self.results.history)
                self.results.to_excel(file)
                file.close()
        else:
            print(f"Save Staus: {saveOutput}, skipping")
    
    def renderModel(self):
        self.modelController.renderModel()