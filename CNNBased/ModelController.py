import os
from datetime import datetime

import cv2
import keras
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
from keras.utils.vis_utils import model_to_dot

from CNNBased.ImageManager import ImageManager
from CNNBased.layers.layerLists import K4S2MP, K4S2MP_GRAY, K4S4, K4S4_GRAY

class ModelController:
    def __init__(self, layers):
        self.layers = layers
        self.model = keras.Sequential(
            layers= self.layers
        )

    def getModel(self):
        return self.model

    def getLayers(self):
        return self.layers

    def renderModel(self) -> None:
        svg = model_to_dot(self.model, show_shapes = True).create(prog='dot', format='svg')

        cv2.imshow('CNN Model Layers',svg )
        cv2.waitkey(0)
        cv2.destroyAllWindows()
