import cv2
import keras
import numpy as np
from keras.utils.vis_utils import model_to_dot
from keras import layers,losses
from threading import Thread

from CNNBased.layers.layerLists import K4S2MP, K4S2MP_GRAY, K4S4, K4S4_GRAY

class ModelController:
    def __init__(self, layers):
        self.layers = layers
        self.model = keras.Sequential(
            layers= self.layers
        )

    def getModel(self, compiled=False):
        if compiled:
            self.model.compile(
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
        return self.model

    def getLayers(self):
        return self.layers

    def renderModel(self) -> None:
        def render():
            summary = self.model.summary()
            print(summary)
            render = model_to_dot(
                self.model, 
                show_shapes = True,
            ).create(
                format='png'
            )
            array = np.fromstring(render,dtype='uint8')
            img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
            cv2.imshow('Model Layers Render', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        th = Thread(target=render)
        th.start()