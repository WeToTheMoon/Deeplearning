import os
from datetime import datetime

import sys
import pandas as pd
from keras import layers, losses

from CNNBased.ImageManager import ImageManager
from CNNBased.ModelController import ModelController
from CNNBased.layers.layerLists import K4S2MP, K4S2MP_GRAY, K4S4, K4S4_GRAY, Testing_Layer, KDESCENDINGS3_GRAY, KDESCENDINGS3, LOW_DENSE


class CNN:
    def __init__(self) -> None:
        self.modelController = ModelController(LOW_DENSE)
        self.model = self.modelController.getModel(compiled=True)

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
            f"Save Output: {None} \n"
            f"Save Filename: {None} \n"
        )

        imageManager = ImageManager(path)
        xTRAIN, xVAL, yTRAIN, yVAL = imageManager.getImages()

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