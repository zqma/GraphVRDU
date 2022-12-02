from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
dataset = load_dataset("rvl_cdip")


subset = dataset["test"]
Dir = "Data/"
label = []
for index in range(subset.shape[0]):
    try:
        data = subset[index]
        image = data["image"]#Image.open(data["image"])
        image.convert('RGB').save( Dir + str(index) + '_.jpg')
        label.append([index, data["label"]])
    except UnidentifiedImageError:
        continue
    
np.savetxt(Dir + "label.txt", label)
