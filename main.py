from CNNBased.CNN import CNN
import sys

# Construct the CNN
CNN = CNN()

# Start the CNN # delete the path if run on Zachs's pc
try:
    sys.argv[1]
    if sys.argv[1].lower() in ["save", "yes", "true", "y"]:
        CNN.start(path=r'C:\Users\achan\Documents\Coding\Python\ZachML\Dataset', saveOutput=True)
    else:
        CNN.start(path=r'C:\Users\achan\Documents\Coding\Python\ZachML\Dataset', saveOutput=False)
except:
    CNN.start(path=r'C:\Users\achan\Documents\Coding\Python\ZachML\Dataset', saveOutput=False)