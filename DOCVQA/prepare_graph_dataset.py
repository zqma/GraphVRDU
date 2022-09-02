from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import TensorDataset, DataLoader, Dataset
from format_input import preprocess_gnn
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import json
import h5py
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name,  output_hidden_states=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.cuda()
model.eval()
    
normalized_levenshtein = NormalizedLevenshtein()



def load_qa(data, split="train"):
    qa_data = []
    with open(data, "r") as f:
        content = json.load(f)

    for qa in content:
        
        img = qa["img"]
        img = Image.open(split + "/" + "documents/"+ img)
        img_size = img.size 
        
        
        
        context = qa["context"]
        question = qa["question"]
        segments = qa["segments"]
        word_bbx = qa["word_bbx"]
        seg_bbx = qa["seg_bbx"]
        sub_graph_ids = qa["sub_graph_loc"]
        
        word_bbx = [item for subbbx in word_bbx for item in subbbx]
        word_bbx = [[box[0], box[1], box[4], box[5]] for box in word_bbx ]
        seg_bbx = [[box[0], box[1], box[4], box[5]] for box in seg_bbx]
         
        if qa["found_answer"]:
            qa_data.append({"question":question, "context": context, "segments": segments, "ans_loc": qa["ans_loc"], "answer":qa["found_answer"], "word_bbox": word_bbx,                                           "seg_bbx":seg_bbx, "sub_graph_loc": sub_graph_ids,"img_size":img_size, "question_id":qa["question_id"]})
                        
    return qa_data

        
        

#Embedding, the output of BERT of question+context: 512 x 768
#Start_pos: the start of ans: int
#end_pos: the end of ans: int
#word_bbox: the bbox coordinate of each token: 512 x 4
#seg_bbox: the bbox of each segment
#seg_ids: the id of the segment that each token belongs to: looks like [0000, 1111, 22222,...]    
#answer: the string that we want to generate    
def generate_embedding(data, split):
    bs = 1
    #for loc in data[0]["sub_graph_loc"]:
    #    print(data[0]["context"][loc[0]: loc[1]])
    
    data = [preprocess_gnn(sample, tokenizer) for sample in data]
    #for loc in data[0]["seg_ids"]:
        #print( tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens( data[0]["input_ids"][loc[0]: loc[1]+1] )))
    
    
    #assert 1==0
    input_ids = [sample["input_ids"] for sample in data]
    attention_mask = [sample["attention_mask"] for sample in data]
    token_type_ids = [sample["token_type_ids"] for sample in data]
    
    tensor_input_ids = torch.tensor(input_ids).to(torch.int64)
    tensor_token_ids = torch.tensor(token_type_ids).to(torch.int64)
    tensor_attention = torch.tensor(attention_mask).to(torch.int64)
    
    dataset = TensorDataset(tensor_input_ids, tensor_token_ids, tensor_attention)
    dataloader = DataLoader(dataset, batch_size=bs)
    
    
    start_pos = [sample["start_positions"] for sample in data]
    end_pos = [sample["end_positions"] for sample in data]
    answer = [sample["answer"] for sample in data]
    word_bbox = [sample["bbox"] for sample in data]
    seg_bbox = [sample["seg_bbx"] for sample in data]
    seg_ids = [sample["seg_ids"] for sample in data]
    q_ids = [sample["question_id"] for sample in data]
    
    Emb = []
    f = h5py.File('graph_ele' + "_" + split+'.hdf5','w')
    for i, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_token_type, b_input_mask = batch 
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=b_token_type,
                        attention_mask=b_input_mask)
            hidden_states = outputs[2]
            embedding = hidden_states[-1].cpu().numpy()
            
            #Emb.append(embedding)
            
    #Emb = np.concatenate(Emb, axis=0)    
    
    
        batch_gnn_input = {"node_embedding": embedding, "answer": answer[i*bs:(i+1)*bs], "word_coordinate": word_bbox[i*bs:(i+1)*bs], "seg_coordinate": seg_bbox[i*bs:(i+1)*bs],                                  "seg_ids": seg_ids[i*bs:(i+1)*bs], "ans_start": start_pos[i*bs:(i+1)*bs], "ans_end": end_pos[i*bs:(i+1)*bs]}        
        
        grp=f.create_group(str(q_ids[i]))
        for k,v in batch_gnn_input.items():
            grp.create_dataset(k, data=v)
    
    
if __name__ == "__main__":
    train_data = "train_annotations.json"
    test_data = "val_annotations.json"
    
    print("begin loading json data")
    train_qa = load_qa(train_data, "train")
    
    #test_qa  = load_qa(test_data, "val")
    print("finish loading json data")
    
    print("generate and save GNN input")
    generate_embedding(train_qa, "train")
    print("Done !!!")
    #sample = preprocess_gnn(train_qa[0], tokenizer)
    
    
  