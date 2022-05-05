from datetime import datetime
from email.policy import default
from CNNBased.CNN import CNN
import argparse
from CNNBased.ImageManager import ImageManager
import sys

parser = argparse.ArgumentParser(description='Save results to .xlsx file',prog='main.py')
parser.add_argument("-save","-s" ,default="false", type=str, help=f'Save file to .xlxs file or not',choices=["true", "false"], required=False)
args = parser.parse_args()

# Construct the CNN
CNN = CNN()

# Start the CNN # delete the path if run on Zachs's pc

if args.save == "true":
    CNN.start(path=r'Dataset', saveOutput=True)
else:
    CNN.start(path=r'Dataset', saveOutput=False)