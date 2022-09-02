###this is a library to process qa input for BERT and GNN
###in general, preprocess function tokenize the question and context, expand function will check the sub tokens and expand the tag
from transformers import AutoTokenizer
import numpy as np

def iden_sub(encoded):
    desired_output = []
    for word_id in encoded.word_ids():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            tokens = [i for i in range(start, end)]
            if len(desired_output) == 0 or desired_output[-1] != tokens:
                desired_output.append(tokens)
    return desired_output

def preprocess_gnn(sample, tokenizer, L=512):
    
    question = sample["question"]
    context = sample["context"]
    ans_loc = sample["ans_loc"]
    bbox = sample["word_bbox"]
    img_size = sample["img_size"]
    sub_graph_loc = sample["sub_graph_loc"]
    
    q_tokens = tokenizer(question, max_length=25, truncation=True, padding = "max_length")
    a_tokens = tokenizer(context, max_length=L-len(q_tokens.input_ids)+1, truncation=True)
    
    sample_ = align_gnn(q_tokens, a_tokens, context, ans_loc, sub_graph_loc, bbox, img_size, L, tokenizer)
    
    sample_["answer"] = sample["answer"]
    sample_["seg_bbx"] = [[0,0,0,0]] + sample["seg_bbx"]
    sample_["question_id"] = sample["question_id"]
    return sample_
  
def preprocess_bert(sample, tokenizer, L=512):
    question = sample["question"]
    answer = sample["context"]
    char_loc = sample["ans_loc"]
    bbox = sample["bbox"]
    img_size = sample["img_size"]
    
    
    q_tokens = tokenizer(question, max_length=25, truncation=True, padding="max_length")#fix the length of questions
    a_tokens = tokenizer(answer, max_length=L-len(q_tokens.input_ids)+1, truncation=True)
    sample_ = align_bert(q_tokens, a_tokens, char_loc, bbox,img_size, L)
    sample_["answer"] = sample["answer"]
    return sample_
    
def align_gnn(q_tokens, a_tokens, context,char_loc, sub_graph_loc,bbox, img_size, max_L=512, tokenizer=None):
    l_q, l_a = len(q_tokens.input_ids), len(a_tokens.input_ids)
    
    #token_sub = iden_sub(a_tokens)
    L = max_L
    
    token_type_ids = make_token_type_ids(l_q, l_a, L)
    input_ids = q_tokens.input_ids + a_tokens.input_ids[1:] + [0]*(L+1-l_q-l_a)
    attn_mask = make_attn_mask(l_q, l_a, L)
    
    #seg_ids = make_seg_ids(np.asarray(input_ids))
    new_bbox = make_bbox(bbox, context, l_q, L, tokenizer)
    new_bbox = [normalize_bbox(box, img_size[0], img_size[1]) for box in new_bbox]
    
    ###get the token position of answer
    start_position = a_tokens.char_to_token(char_loc[0]) 
    end_position = a_tokens.char_to_token(char_loc[1]-1)
    
    if start_position is None:
        start_position = L
    else:
        start_position = start_position + l_q - 1
    if end_position is None:
        end_position = L
    else:
        end_position = end_position + l_q -1
          
    qa_sample = {"input_ids": input_ids, "attention_mask": attn_mask, "token_type_ids": token_type_ids, "seg_ids": make_sub_graph_seg(sub_graph_loc, a_tokens, l_q-1, L),                                "bbox":new_bbox, "start_positions": start_position, "end_positions": end_position}
    return  qa_sample

    
def align_bert(q_tokens, a_tokens, char_loc, bbox, img_size, max_L=512):
    l_q, l_a = len(q_tokens.input_ids), len(a_tokens.input_ids)
    new_tag = [0]* l_q
    new_bbox = [[0, 0, 0, 0] for i in range(l_q)]
    #token_sub = iden_sub(a_tokens)
    L = max_L
    
    token_type_ids = make_token_type_ids(l_q, l_a, L)
    input_ids = q_tokens.input_ids + a_tokens.input_ids[1:] + [0]*(L+1-l_q-l_a)
    attn_mask = make_attn_mask(l_q, l_a, L)
    
    new_bbox = [normalize_bbox(box, img_size[0], img_size[1]) for box in new_bbox]
    
    
    start_position = a_tokens.char_to_token(char_loc[0]) 
    end_position = a_tokens.char_to_token(char_loc[1]-1)
    
    if start_position is None:
        start_position = L
    else:
        start_position = start_position + l_q - 1
    if end_position is None:
        end_position = L
    else:
        end_position = end_position + l_q -1
    qa_sample = {"input_ids": input_ids, "attention_mask": attn_mask, "token_type_ids": token_type_ids, "start_positions": start_position, "end_positions": end_position, "bbox"                          :new_bbox}
    return  qa_sample

def make_token_type_ids(l1, l2, L):

    return l1*[0] + (l2-1)*[1] + (L+1-l1-l2)*[0]
    
def make_attn_mask(l1, l2, L):

    return (l1+l2-1)*[1] + (L+1-l1-l2)*[0]
    
def normalize_bbox(bbox, width, height):
    bbox[0] = int(1000*bbox[0]/width)
    bbox[1] = int(1000*bbox[1]/height)
    bbox[2] = int(1000*bbox[2]/width)
    bbox[3] = int(1000*bbox[3]/height)
    
    return bbox
def make_sub_graph_seg(sub_graph_loc, encoding, shift, L):
    seg_ids = []
    for loc in sub_graph_loc:
        start = encoding.char_to_token(loc[0]) 
        end = encoding.char_to_token(loc[1]-1) 
        if start==None: 
            start = L
        else:
            start += shift
        if end==None: 
            end = L
        else:
            end += shift

        seg_ids.append([start, end])
    return seg_ids
    
def make_bbox(bbox, context, l_q, L, tokenizer):
    new_bbox = [[0, 0, 0, 0] for i in range(l_q)]
    
    for word, box in zip(context.split(), bbox):
        num_sub_token = len(tokenizer(word).input_ids)-2
        new_bbox += [box]*num_sub_token
    
    if len(new_bbox)<L:
        new_bbox += [[0,0,0,0] for i in range(L-len(new_bbox))]
    else:
        new_bbox = new_bbox[:L]    
    return new_bbox

#def make_seg_ids(input_ids, sep=102, padding=0):
#    seg_ids = []
#    
#    segs = np.split(input_ids, np.argwhere(input_ids==sep).flatten())
#    last = len(segs)-1
#    for i, seg in enumerate(segs):
#        if i == 0:
#          seg_ids += [i]*(len(seg) +1)
#        elif i == last:
#          x= -1 if seg.all()==padding else i
#          seg_ids += [x]*(len(seg)-1) 
#        else:
#          seg_ids += [i]*(len(seg)) 
#        
#    return seg_ids

#this is for testing the functions   
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    answer = ["I like cheeseburger and cheesefries", "I like icecream"]
    question = "What do I like?"
    
    tag = [[0,0,1,1,1], [0,0,1]]
    max_L =30
    
    sample = {"question":question, "context":answer, "tag":tag}
    x = preprocess_gnn(sample, tokenizer, max_L)
    
    
    
    


    