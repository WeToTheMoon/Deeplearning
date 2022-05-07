

from email.mime import base
import os
from pydoc import doc
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

    def refactorImage(self, image, doCropping:bool = False):
        if doCropping:
            cropped_image = self.cropImage(image)
            img = cv2.resize(cropped_image, (256, 256))
        else:
            img = cv2.resize(image, (256, 256))

        return img

    def cropImage(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(thresh, dtype=np.uint8)
        cv2.drawContours(mask, [big_contour], 0, 255, -1)
        x,y,w,h = cv2.boundingRect(big_contour)
        img_crop = image[y:y+h, x:x+w]
        return img_crop

    def getImages(self):
        try:
            self.__dataset
        except:
            sys.exit("Dataset not loaded")

        for type in  self.__cancerTypes:
            print(f"Loading and pre-processing {type} images")
            folderPath = os.path.join(self.__dataset, type)
            for j in tqdm(os.listdir(folderPath)):
                img = cv2.imread(os.path.join(folderPath, j))
                img = self.refactorImage(image=img, doCropping=False)
                self.__imageList.append(img)
                self.__typeList.append(type)

        self.__imageList = np.array(self.__imageList)
        self.__typeList = np.array(self.__typeList)

        print(
            
            f"Caching Images in ram \n"
            f"Total Size in bytes: {self.__imageList.nbytes + self.__typeList.nbytes}"
        )

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

    def testImages():
        pass

# x = ImageManager(r'Dataset')

# img1 = cv2.imread(r'C:\Users\achan\Documents\Coding\Python\ZachML\Dataset\glioma\721image.png')
# img2 = cv2.imread(r'C:\Users\achan\Documents\Coding\Python\ZachML\Dataset\glioma\829image.png')
# img3 = cv2.imread(r"C:\Users\achan\Documents\Coding\Python\ZachML\Dataset\glioma\1615image.png")
# img4 = cv2.imread(r"C:\Users\achan\Documents\Coding\Python\ZachML\Dataset\meningioma\552image.png")

# crop_img1, crop_img2, crop_img3, crop_img4 = x.refactorImage(img1, doCropping=True), x.refactorImage(img2, doCropping=True), x.refactorImage(img3, doCropping=True), x.refactorImage(img4, doCropping=True)


# cv2.imshow('testingImage 1_crop', crop_img1)
# cv2.imshow('testingImage 2_crop', crop_img2)
# cv2.imshow('testingImage 3_crop', crop_img3)
# cv2.imshow('testingImage 4_crop', crop_img4)

# cv2.imshow('testingImage 1', img1)
# cv2.imshow('testingImage 2', img2)
# cv2.imshow('testingImage 3', img3)
# cv2.imshow('testingImage 4', img4)

# cv2.waitKey(0)
# cv2.destroyAllWindows()