import argparse

from CNNBased.CNN import CNN

parser = argparse.ArgumentParser(description='Save results to .xlsx file',prog='main.py')
parser.add_argument("-save","-s" ,default="false", type=str, help=f'Save results to .xlxs file or not',choices=["true", "false"], required=False)
args = parser.parse_args()

# Construct the CNN
CNN = CNN()

# Start the CNN

CNN.start(path=r'Dataset')
# CNN.renderModel()
if args.save == "true":
    CNN.saveResults(saveOutput=True)
else:
    CNN.saveResults(saveOutput=False)

