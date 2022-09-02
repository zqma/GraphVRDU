from transformers import AutoTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import TensorDataset, DataLoader, Dataset
from format_input import preprocess_bert
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import torch
import json
from PIL import Image
  

#model_name = "tiennvcs/bert-large-uncased-finetuned-docvqa"
model_name = "bert-base-uncased"
#model_name = "microsoft/layoutlmv2-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
normalized_levenshtein = NormalizedLevenshtein()
class QA_Dataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        for i in range(len(data)):
            
            sample = preprocess_bert(data[i], tokenizer)
            
            self.data.append(sample)
            
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        sample["input_ids"] = torch.tensor(sample["input_ids"]).to(torch.int64)
        sample["token_type_ids"] = torch.tensor(sample["token_type_ids"]).to(torch.int64)
        sample["attention_mask"] = torch.tensor(sample["attention_mask"]).to(torch.int64)
        sample["start_positions"] = torch.tensor(sample["start_positions"]).to(torch.int64)
        sample["end_positions"] = torch.tensor(sample["end_positions"]).to(torch.int64)
        sample["bbox"] = torch.tensor(sample["bbox"]).to(torch.int64)
        return sample

def extract_answer(context, tag):
    words = context.split()
    
    return " ".join([words[i] for i, label in enumerate(tag) if label==1])

def sort_seg(context, segs):
    keys = [seg["bbox"][:2] for seg in segs] #extract the leftupper point of the bbox for sorting
    
    key_segs = [(keys[i],segs[i], context[i]) for i in range(len(keys))] #zip key and value for sorting
    
    key_segs.sort(key=lambda x: x[0][1]) #sort by y1
    key_segs.sort(key=lambda x: x[0][0]) #sort by x1
    
    _, segs, context = zip(*key_segs)
    return context, segs

def load_qa(data, split="train"):
    qa_data = []
    with open(data, "r") as f:
        content = json.load(f)

    for qa in content:
        
        #img = split + "/" + "documents/"+img
        #img = Image.open(img)
        #img_size = img.size #fix later
        img_size = 1000, 1000
        
        
        context = qa["context"]
        question = qa["question"]
        
        word_bbx = qa["word_bbx"]
        word_bbx = [item for subbbx in word_bbx for item in subbbx]
        word_bbx = [[box[0], box[1], box[4], box[5]] for box in word_bbx]
        
            
        ans_span = qa["ans_span"]
         
        if qa["found_answer"]:
            qa_data.append({"question":question, "context": context, "ans_loc": qa["ans_loc"] , "answer":qa["found_answer"], "bbox": word_bbx, "img_size":img_size})
                        
    return qa_data
    

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    
    for d in data_loader:
        
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        bbox = d["bbox"].to(device)
        start_positions = d["start_positions"].to(device)
        end_positions = d["end_positions"].to(device)
    
         
        outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                start_positions = start_positions,
                end_positions = end_positions
                )
      
        loss = outputs[0]
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
    return  np.mean(losses)    
    
def eval_model(model, data_loader, device, max_ans_len=20):
    model = model.eval()
    losses = []
    answer_pred = []
    answer_gt = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            bbox = d["bbox"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            start_positions = d["start_positions"].to(device)
            end_positions = d["end_positions"].to(device)
            
            outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    token_type_ids = token_type_ids,
                    start_positions = start_positions,
                    end_positions = end_positions
                    )
            start_pred = torch.argmax(outputs['start_logits'], dim=1).clone().detach().cpu().numpy()
            end_pred = torch.argmax(outputs['end_logits'], dim=1).clone().detach().cpu().numpy()
            
            answer_gt.extend(d["answer"])
            for s, e, input_id in zip(start_pred, end_pred, d["input_ids"]):
                e = min(e, s+ max_ans_len)
                #if e<s: e = s
                answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_id[s:e+1]))
                answer_pred.append(answer)
            
    return  answer_pred, answer_gt

def compute_anls(pred, gt):
    res = []
    for p, g in zip(pred, gt):
        res.append(normalized_levenshtein.similarity(p, g))
    return np.mean(res)
    

if __name__ == "__main__":
    train_data = "train_annotations.json"
    test_data = "val_annotations.json"
    
    
    
    train_qa = load_qa(train_data, split="train")
    test_qa = load_qa(test_data, split="val")    
    train_dataset = QA_Dataset(train_qa)
    test_dataset = QA_Dataset(test_qa)
    
    
    
    
    batch_size = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    optim = AdamW(model.parameters(), lr=5e-5)
    
    
    EPOCHS = 2
    history = defaultdict(list)
    best_ce = float("inf")
    tolerance, patience = 0, 2
   
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print('-' * 10)
        train_loss = train_epoch(model, train_loader,optim, device )
        print(f'Train loss {train_loss} ')
        history['train_loss'].append(train_loss)
        
        if train_loss < best_ce:
            torch.save(model.state_dict(), 'BERT_QA.bin')
            best_ce = train_loss
            tolerance = 0
        else:
            tolerance += 1
            if tolerance > patience:
                break
        
        if epoch % 1 == 0:
            pred_answer, gt_answer = eval_model(model, test_loader, device)
            anls = compute_anls(pred_answer, gt_answer)
            print(f"ANLS on test set is {anls}")
        
    with open('predictions.txt', 'w') as f:
        for line in zip(gt_answer, pred_answer):
            f.write(f"{line}\n")
    
    