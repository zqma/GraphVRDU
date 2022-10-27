import numpy as np
import json
import os
import glob
import re
from collections import defaultdict
qas = defaultdict(list)

##sort the orc output in top-left to bottom-right manner
def sort_ocr(words, word_bbox, seg_bbox):
    keys = [seg for seg in seg_bbox] #extract the leftupper point of the segment bbox for sorting
    
    #word_bbox = [[x[0], x[1], x[4], x[5]] for x in word_bbox] #extract the leftupper and rightlower points of the bbox for sorting
    
    key_words = [(keys[i], words[i], word_bbox[i]) for i in range(len(keys))] #zip key and value for sorting
    
    key_words.sort(key=lambda x: x[0][0]) #sort by x1
    key_words.sort(key=lambda x: x[0][1]) #sort by y1
    

    seg_bbx, context, word_bbox = zip(*key_words)
    
    
    
    return context, word_bbox, seg_bbx

##load all the questions and group them by docid    
def load_questions(questions):
    with open(questions, "r") as f:
        question_data = json.load(f)
    
    ###read question json file       
    for sample in question_data["data"]:
        
        docid = sample["image"].split("/")[1]   
        qa = {}
        
        qa["question"] = sample["question"]
        
        if index<2: qa["answer"] = list(set([ans.lower() for ans in sample["answers"]]))
        qa["id"] = sample["questionId"]
        qas[docid].append(qa)      
    return qas
    
def check_substring(ans, context, pos):
    res = len(context[:pos].split())
    start = res if context[pos-1] == " " else res-1
    end = start + len(ans.split())
                                
    ans_span[start:end] = 1
    found_ans = " ".join([x for i,x in zip(ans_span, context.split()) if i==1])  
                            
def transform(string):
    new_string = ""
    specials = ["(", ")", "[", "]", "{", "}", "+", "^", "*", "?"]
    
    for c in string:
        if c in specials: 
            c = '\\' + c
        new_string += c
        new_string += "[\s\.]*"
    
    return re.compile(new_string )
    
def get_seg_id(segs):
    res = 0
    sub_graph_loc = []
    start, end = 0, 0
    for i, seg in enumerate(segs):
        if i == 0:
            start += 0
            end += len(seg)
        else:
            start = end+1
            end = start + len(seg)
        sub_graph_loc.append([start, end])
        
        
    return sub_graph_loc
if __name__ == "__main__":
    folders = ["train", "val", "test"]
    index = 1
    dataset = folders[index] + "/" 
 
    
    questions = dataset + folders[index] + "_v1.0.json"
    
    qas = load_questions(questions) 
    
    ###read doc ocr results
    qa_squad_format = []
    
    for d, qa_list in qas.items():
        
        ocr = dataset + "ocr_results/"+ d.replace(".png", ".json")
        with open(ocr, "r") as f:
            content = json.load(f)
            
        words = []
        word_bbx = []
        seg_bbox = []    
        for seg in content["recognitionResults"][0]["lines"]:
            seg_bbox.append(seg["boundingBox"])
            ##extract all the words and corresponding bbox from ocr
            
            words.append(" ".join([x["text"].lower() for x in seg["words"]]))
            word_bbx.append([x["boundingBox"] for x in seg["words"]])
            
        ##sort the results  
        assert len(words) == len(word_bbx)
        if len(words) == 0: continue 
        words, word_bbx, seg_bbx = sort_ocr(words, word_bbx, seg_bbox)   
        context = " ".join(words)
            
        #a document might be associated with multiple questions    
        for qa in qa_list:
            s, e = -1, -1   
            #check if answer is in the segment
            exist = False
            q = qa["question"].lower()
            if index < 2: 
                #change the logic, read question first, then  load doc
                for ans in qa["answer"]:
                    ans_span = np.zeros(len(context.split()))
                    found_ans = None
                    #pos = context.find(ans)
                    
                    #check for all the occurances of ans
                    #allowe space between characters e.g. "dr. J" vs "dr.J"
                    pattern = transform(ans)
                        
                    matches = [[m.start(), m.end()] for m in re.finditer(pattern, context)]
                    #this will include false answer like "1" will be found in "1900"    
                    valid = False  
                    for match in matches: 
                         
                        s, e = match
                        matched_ans = context[s:e]
                        
                        res = len(context[:s].split())
                        start = res if (context[s-1] == " ") or (s==0) else res-1
                        end = start + len(matched_ans.split())
                                
                        ans_span[start:end] = 1
                        found_ans = " ".join([x for i,x in zip(ans_span, context.split()) if i==1])  
                        valid = True
                            
                        ##filter noisy answer like "1" vs "1900"    
                        
                        if (end == start+1) and re.sub('[^A-Za-z0-9]+', '', ans) != re.sub('[^A-Za-z0-9]+', '', found_ans): 
                            ans_span[:] = 0
                            valid = False
                            found_ans = None
                            start, end = -1, -1
                           
                        if valid: break             
                    if valid: break
                                  
            else:
                pass
            sub_graph_loc = get_seg_id(words)       
            qa_pair = {"ans_span": ans_span.tolist(), "context": context, "word_bbx": word_bbx, "seg_bbx":seg_bbx ,"segments":words, "sub_graph_loc": sub_graph_loc, "gt_answer": ans,                             "found_answer": found_ans, "question_id": qa["id"], "question": q, "ans_loc": [s, e-1], "img": d}

            qa_squad_format.append(qa_pair)
     
    #print(qa_squad_format[1])   
    with open(folders[index] + "_annotations.json", "w") as f:
        json.dump(qa_squad_format, f)
    print("parsing done!!!")
    """ 
    for i in range(28,29):
        print("#"*10 + str(i)+"#"*10)
        sample = qa_squad_format[i]
        if sample["sub_graph_loc"]:
            print(sample["segments"])
            for loc in sample["sub_graph_loc"]:
                print(sample["context"][loc[0]:loc[1]+1])
            #print(sample["question"], " gt:",sample["gt_answer"], " found:", sample["found_answer"], sample["ans_loc"])
            #print(sample["found_answer"] )
    """