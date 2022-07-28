from transformers import AutoTokenizer, AutoModel, ViTFeatureExtractor, ViTModel
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import numpy as np
import torch
import cv2
import sys
import json
import re
import h5py

lm = "bert-base-uncased"
vm = "google/vit-base-patch16-224-in21k"
def preprocess(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    #cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return img

path_to_dataset = "annotation.json"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(path_to_dataset, 'rb') as fp:
    data = json.load(fp)

####generate tokens
tokenizer = AutoTokenizer.from_pretrained(lm)
#docs = {}

text_encoder = AutoModel.from_pretrained(lm, output_hidden_states=True)
text_encoder.cuda()
text_encoder.eval()

img_feature_extractor = ViTFeatureExtractor.from_pretrained(vm)
image_encoder = ViTModel.from_pretrained(vm)


res = []
f = h5py.File('graph_ele.hdf5','w') 


#print("tokenization starts")
i = 0
for img, anno in data.items():
    i += 1
    print(i)
    if not anno: continue
    doc = {}
    #load doc image for cropping
    image = cv2.imread(img)
    image = preprocess(image)
    image = image[..., ::-1]  
    
    #unpack values
    batch = list(anno.keys())
    values = list(anno.values())
    
    coors, labels = list(zip(*values))
    
    #generate crops
    
    crops = [image[max(int(x[1]-5),0):int(x[3]+5), max(int(x[0]-5),0):int(x[2]+5)] for x in coors]
    #for i, c in enumerate(crops):
        #Image.fromarray(c).save( str(i) + "_.jpg")
       
        
    batch = [re.sub('\W+',' ', string) for string in batch]
    batch_tokens = tokenizer(batch, truncation=True, padding='max_length', max_length=128)
    
    doc["input_ids"] = torch.tensor(batch_tokens["input_ids"]).to(torch.int64)
    doc["token_type_ids"] = torch.tensor(batch_tokens["token_type_ids"]).to(torch.int64)
    doc["attention_mask"] = torch.tensor(batch_tokens["attention_mask"]).to(torch.int64)
    doc["crops"] = crops
    doc["label"] = labels[0]
    doc["coordinates"] = coors
    
    
    sample = {}
    batch = [doc["input_ids"], doc["token_type_ids"], doc["attention_mask"]]
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_token_type, b_input_mask = batch
        
    with torch.no_grad():
            
        outputs = text_encoder(b_input_ids, token_type_ids=b_token_type,
                        attention_mask=b_input_mask)
        hidden_states = outputs[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        #res.append(sentence_embedding.cpu().numpy())
        
    sample["text_emb"] = sentence_embedding.cpu().numpy()
    
    #print("------")
    #for c in content["coordinates"]:
        #print(c)
    vm_inputs = img_feature_extractor(doc["crops"], return_tensors="pt")

    with torch.no_grad():
        outputs = image_encoder(**vm_inputs)
        last_hidden_state = outputs.last_hidden_state
        image_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    sample["img_emb"] = image_embedding.cpu().numpy()
    
    sample["coordinates"] = doc["coordinates"]
    sample["label"] = doc["label"]
    
    
    grp=f.create_group(img)
    for k,v in sample.items():
        grp.create_dataset(k, data=v)
    
    #docs[img] = doc
   
    
#print("tokenization done")


#print("begin loading tokens")
#input_ids = np.loadtxt(path_to_dataset+"input_ids.txt")
#token_type_ids = np.loadtxt(path_to_dataset+"token_type_ids.txt")
#attention_mask = np.loadtxt(path_to_dataset+"attention.txt")
#assert input_ids.shape == token_type_ids.shape
#print("tokens loading done")
    
#tensor_input_ids = torch.tensor(input_ids).to(torch.int64)
#tensor_token_ids = torch.tensor(token_type_ids).to(torch.int64)
#tensor_attention = torch.tensor(attention_mask).to(torch.int64)
    
#dataset = TensorDataset(tensor_input_ids, tensor_token_ids, tensor_attention)
#dataloader = DataLoader(dataset, batch_size=bacth_size)
    

#the commented part will generate embeddings of ocr results

"""
for img, content in docs.items():
    
    sample = {}
    batch = [content["input_ids"], content["token_type_ids"], content["attention_mask"]]
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_token_type, b_input_mask = batch
        
    with torch.no_grad():
            
        outputs = text_encoder(b_input_ids, token_type_ids=b_token_type,
                        attention_mask=b_input_mask)
        hidden_states = outputs[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        #res.append(sentence_embedding.cpu().numpy())
        
    sample["text_emb"] = sentence_embedding.cpu().numpy()
    
    #print("------")
    #for c in content["coordinates"]:
        #print(c)
    vm_inputs = img_feature_extractor(content["crops"], return_tensors="pt")

    with torch.no_grad():
        outputs = image_encoder(**vm_inputs)
        last_hidden_state = outputs.last_hidden_state
        image_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    sample["img_emb"] = image_embedding.cpu().numpy()
    
    sample["coordinates"] = content["coordinates"]
    sample["label"] = content["label"]
    
    dataset[img] = sample

np.save("graph_elements.npy", dataset)
"""
#np.save(filename+str(split)+".npy", np.stack(res, axis=0))    
