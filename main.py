from datetime import datetime
from CNNBased.CNN import CNN
import argparse
import sys

parser = argparse.ArgumentParser(description='Save results to .xlsx file',prog='main.py')
parser.add_argument("-save","-s" ,default=False, type=bool, help=f'Save file to .xlxs file or not',choices=[True, False])
parser.add_argument("-filename", "-f", default=rf'results\cnn\Results {datetime.now().strftime(r"%Y_%m_%d-%I%M%S_%p")}.xlsx', type=str, help=f'File name', required=False)
args = parser.parse_args()

# Construct the CNN
CNN = CNN()

# Start the CNN # delete the path if run on Zachs's pc

if args.save:
    CNN.start(path=r'Dataset', saveOutput=True, filename=args.filename)
else:
    CNN.start(path=r'Dataset', saveOutput=False, filename=args.filename)