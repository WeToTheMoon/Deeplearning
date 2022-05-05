

import os
import sys

import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ImageManager:
    def __init__(self, datasetPath: str):
        self.__dataset = datasetPath
        self.__cancerTypes = ['glioma', 'meningioma', 'pituitary']
        self.__typeList, self.__imageList,  = [], []
        self.__y_train_new, self.__y_test_new = [], []

    def ver(self):
        print("hi")

    def images(self):
        try:
            self.__dataset
        except:
            sys.exit("Dataset not loaded")

        for type in  self.__cancerTypes:
            print(f"Loading and pre-processing {type} images")
            folderPath = os.path.join(self.__dataset, type)
            for j in tqdm(os.listdir(folderPath)):
                img = cv2.imread(os.path.join(folderPath, j))
                img = cv2.resize(img, (256, 256))
                self.__imageList.append(img)
                self.__typeList.append(type)

        print("Caching Images")

        self.__imageList = np.array(self.__imageList)
        self.__typeList = np.array(self.__typeList)

        xTRAIN, xVAL, yTRAIN, yVAL = train_test_split(self.__imageList, self.__typeList, test_size=0.2, random_state=14)

        for i in yTRAIN:
            self.__y_train_new.append(self.__cancerTypes.index(i))

        yTRAIN = to_categorical(self.__y_train_new)

        for i in yVAL:
            self.__y_test_new.append(self.__cancerTypes.index(i))

        yVAL = to_categorical(self.__y_test_new)

        xTRAIN = xTRAIN / 255
        xVAL = xVAL / 255
        
        return xTRAIN, xVAL, yTRAIN, yVAL