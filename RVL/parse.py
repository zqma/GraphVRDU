from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import layoutparser as lp
import cv2
import glob
import warnings
import json
warnings.filterwarnings("ignore") 
#dataset = load_dataset("rvl_cdip")
#data = dataset["train"][1]
#image = data["image"]#Image.open(data["image"])
#image.convert('RGB').save('my.jpg')

Dir = "Data/"
label = np.loadtxt(Dir + "label.txt")


index = [1]#[i for i in range(10)]

model = lp.models.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
  
def preprocess(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    #cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return img
                               
                                 
def prepare_ocr(image, text_blocks):                   
    h, w = image.shape[:2]

    left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
    # The b.coordinates[1] corresponds to the y coordinate of the region
    # sort based on that can simulate the top-to-bottom reading order 
    right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
    right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

    # And finally combine the two lists and add the index
    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
    return text_blocks

def ocr_process(text_blocks, ocr_agent):
    
    for block in text_blocks:
        segment_image = (block
                       .pad(left=10, right=10, top=10, bottom=10)
                       .crop_image(image))
            # add padding in each image segment can help
            # improve robustness 
        
        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)
    

if __name__ == "__main__":
    Annotations = {}        
    ocr_agent = lp.TesseractAgent(languages='eng')
    
    #for filename in glob.glob(Dir + '*.jpg'): #assuming gif
        #image_list.append(filename)
    #print(len(image_list))
    i = 1   
    label = np.loadtxt(Dir+"label.txt")
    label = pd.DataFrame(label, columns=["index", "category"])
    label["index"] = label["index"].astype(int).astype(str)
    label["category"] = label["category"]
    
    
    split_label = train_test_split(
                     label, test_size=0.2, random_state=42, stratify = label["category"])
    
    sub_label = split_label[1]
    
    for i in range(sub_label.shape[0]):
        print(i)
        res = {}    
        image_path = Dir + sub_label["index"].iloc[i] + "_.jpg"
        image = cv2.imread(image_path)
        image = preprocess(image)
        image = image[..., ::-1]              
        layout = model.detect(image)
        #bbox = lp.visualization.draw_box(image, layout) #this will plot and save the image
        #bbox.save(str(i) + "_.jpg")
        text_blocks = lp.Layout([b for b in layout if b.type !='Figure'])
        text_blocks = prepare_ocr(image, text_blocks)
        ocr_process(text_blocks, ocr_agent)
        coordinates = [(x.block.x_1,x.block.y_1, x.block.x_2, x.block.y_2 ) for x in text_blocks._blocks]
        for txt, img, coor in zip(text_blocks.get_texts(),text_blocks.crop_image(image), coordinates):
            if txt.isspace(): continue
            res[txt] = [coor, int(sub_label["category"].iloc[i])]
            #print(txt, end='\n---\n')
            #print(coor)
            #Image.fromarray(img).save( str(j) + "_.jpg")
            #j += 1
        
        Annotations[image_path] =res
    print(Annotations)
    #this will save the output as json file
    with open('annotation.json', 'w') as fp:
        json.dump(Annotations, fp)
        
    
