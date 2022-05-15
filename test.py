import cv2
import pydicom
import numpy as np
from PIL import Image
ds = pydicom.read_file(r"C:\Users\kesch\OneDrive\Desktop\Rebrand\manifest-1636603674498\GLIS-RT\GLI_001_GBM\04-05-2008-NA-MRI BRAIN TUMOR WITH AND WITHOUT CONTRAST-71868\1.000000-REG T1T2-67019\1-1.dcm")
print(ds)