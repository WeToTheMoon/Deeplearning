import argparse

from CNNBased.CNN import CNN

parser = argparse.ArgumentParser(description='Save results to .xlsx file',prog='main.py')
parser.add_argument("-save","-s" ,default="false", type=str, help=f'Save file to .xlxs file or not',choices=["true", "false"], required=False)
args = parser.parse_args()

# Construct the CNN
CNN = CNN()

# Start the CNN # delete the path if run on Zachs's pc
CNN.start(path=r'Dataset')
if args.save == "true":
    CNN.saveResults(path=r'Dataset', saveOutput=True)
else:
    CNN.saveResults(path=r'Dataset', saveOutput=False)
