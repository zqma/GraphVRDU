from datasets import Dataset, DatasetDict,load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoTokenizer
import transformers
import os
import json
import pandas as pd
import pickle
import numpy as np
# disable caching!!
import datasets
datasets.disable_caching()

class DocVQA:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        Dataset({
            features: ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'start_positions', 'end_positions'],
            num_rows: 50
        })
        '''
        # 1. load raw train_test datasets
        self.train_id2doc = self._load_pickle(self.opt.docvqa_pickles + 'train_pickle.pickle')
        self.test_id2doc = self._load_pickle(self.opt.docvqa_pickles + 'val_pickle.pickle')
        self.train_test_dataset = self.get_train_test(train='train',test='val')
        print(self.train_test_dataset)
        del self.train_id2doc
        del self.test_id2doc

        # 2.1 tokenize the words into sequences and positions
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)    #tokenizer
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast) # get sub

        # 2.2 map train and test data to proper features;
        self.train_dataset, self.test_dataset = self.get_dataset('train'), self.get_dataset('test')

        # print(self.test_dataset)
        # doc1 = self.test_dataset[0]
        # print(doc1['input_ids'])

    def get_train_test(self,train,test):
        train = Dataset.from_generator(self._load_QA_pairs, gen_kwargs={'split':train})
        test = Dataset.from_generator(self._load_QA_pairs, gen_kwargs={'split':test})
        return DatasetDict({
            "train" : train , 
            "test" : test 
        })

    def _load_QA_pairs(self,split='val'):
        # from pickle of docs
        if split=='train':
            id2one_doc = self.train_id2doc
        else:
            id2one_doc = self.test_id2doc

        # from json of questions and answers
        file_path = os.path.join(self.opt.docvqa_dir, split, split+'_v1.0.json')
        with open(file_path) as fr:
            data = json.load(fr)
        # df = pd.DataFrame(data['data'])
        for sample in data['data']:
            qID = sample['questionId']
            question = sample['question']
            answers = sample['answers']
            
            # docID = sample['docId'] # e.g.: 14281
            # file name  = ucsf_doc_id + '_' + ucsf-doc_page_no  + suffix
            ucsf_doc_id = sample['ucsf_document_id']   # e.g.,: txpp0227
            ucsf_doc_page = sample['ucsf_document_page_no'] # e.g.,: 10
            docID_page = ucsf_doc_id + '_' + ucsf_doc_page

            if (docID_page not in id2one_doc.keys()) or (not id2one_doc[docID_page]):
                # print(docID_page)
                continue
            # answers = sample['answers'] # e.g.,: ['TRRF Vice President', 'lee a. waller']

            # image value is like, e.g.: documents/txpp0227_10.png
            # image_path = self.docvqa_dir + split + sample['image']

            # append from q-a pairs
            # append from json files: words and boxes, and seg_ids
            words=id2one_doc[docID_page]['tokens']
            boxes = id2one_doc[docID_page]['bboxes']
            image = id2one_doc[docID_page]['image'][0]

            yield {"questionId": qID,'question':question, 'answers':answers, "words": words, "boxes": boxes,
                         "image": image,
                    }


    def get_dataset(self,split='train',shuffle=True):
        '''
        return sth like:
        Dataset({
            features: ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'ans_start_positions', 'ans_end_positions'],
            num_rows: 50
        })
        '''
        # 1. data split return self.prepare_one_doc(self.dataset[split])
        dataset = self.train_test_dataset[split]   # get split
        # 2. feature definition
        features = Features({
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'ans_start_positions': Value(dtype='int64'),
                'ans_end_positions': Value(dtype='int64'),
            })
        # 3. produce dataset
        trainable_dataset = dataset.map(
            self._prepare_one_batch,
            batched=True, 
            batch_size = self.opt.batch_size,
            remove_columns=dataset.column_names,    # remove original features
            features = features,
            # load_from_cache=False,
        ).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        # 4. return dataset
        return trainable_dataset



    def _prepare_one_batch(self,examples, max_length=512):
        # 1. take a batch 
        question = examples['question']
        answers = examples['answers']
        words = examples['words']
        boxes = examples['boxes']

        # 2. encode it
        encoding = self.tokenizer(question, words, boxes, max_length=max_length, padding="max_length", truncation=True, return_token_type_ids=True)

        # 3. next, add start_positions and end_positions
        ans_start_positions = []
        ans_end_positions = []
        
        # for every example in the batch:
        for batch_index in range(len(answers)):
            # print("Batch index:", batch_index)  
            # print(question[batch_index])
            # print(answers[batch_index])
            # print(words[batch_index])
            # 3.1 step1: match answer range in raw word idx []0,1,2,..] to get e.g., [3,5]
            match, ans_word_idx_start, ans_word_idx_end = self._raw_ans_word_idx_range(words[batch_index], answers[batch_index])
            
            # 3.2 step2: match answer range in the sequence e.g., [None, 0,1,2,2,2,3,3,4,5,6,7,7,7, None] to get index range
            if match:
                answer_start_index, answer_end_index = self._ans_index_range(encoding,batch_index,ans_word_idx_start, ans_word_idx_end)
                ans_start_positions.append(answer_start_index)
                ans_end_positions.append(answer_end_index)
                # print("Verifying start position and end position:===")
                # print("True answer:", answers[batch_index])
                # reconstructed_answer = self.tokenizer.decode(encoding.input_ids[batch_index][answer_start_index:answer_end_index+1])
                # print("Reconstructed answer:", reconstructed_answer)
                # print("-----------")
            else:
                cls_index = encoding.input_ids[batch_index].index(self.tokenizer.cls_token_id)
                # print("Answer not found in context")
                # print("-----------")
                ans_start_positions.append(cls_index)
                ans_end_positions.append(cls_index)
        # 3.3 append the ans_start, ans_end_index
        encoding['pixel_values'] = examples['image']   # sometimes, it needs to change it into open Image !!!!!
        encoding['ans_start_positions'] = ans_start_positions
        encoding['ans_end_positions'] = ans_end_positions

        return encoding

    # find the index, of the answer for raw tokens;
    def _subfinder(self, words_list, answer_list):  
        # print('input words:',words_list)
        # print('input ans:',answer_list)
        matches = []
        start_indices = []
        end_indices = []
        for idx, i in enumerate(range(len(words_list))):
            if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
                matches.append(answer_list)
                start_indices.append(idx)
                end_indices.append(idx + len(answer_list) - 1)
        if matches:
            return matches[0], start_indices[0], end_indices[0]
        else:
            return None, 0, 0
    def _raw_ans_word_idx_range(self, words, answers):
        # Match trial 1: try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words]
        for answer in answers:
            match, ans_word_idx_start, ans_word_idx_end = self._subfinder(words_example, answer.lower().split())
            if match:
                break
        return match, ans_word_idx_start, ans_word_idx_end 
    def _ans_index_range(self, batch_encoding,batch_index, answer_word_idx_start, answer_word_idx_end):
        sequence_ids = batch_encoding.sequence_ids(batch_index) # types 0s and 1s
        # Start token index of the current span in the text.
        left = 0
        while sequence_ids[left] != 1:
            left += 1
        # End token index of the current span in the text.
        right = len(batch_encoding.input_ids[batch_index]) - 1
        while sequence_ids[right] != 1:
            right -= 1

        sub_word_ids = batch_encoding.word_ids(batch_index)[left:right+1]
        for id in sub_word_ids:
            if id == answer_word_idx_start:
                break
            else:
                left += 1
        for id in sub_word_ids[::-1]:
            if id == answer_word_idx_end:
                break
            else:
                right -= 1
        # return the result (ans_index_start, ans_index_end)
        return [left,right]
        

    def _load_pickle(self,picke_path):
        with open(picke_path,'rb') as fr:
            res = pickle.load(fr)
        return res



if __name__ == '__main__':
    # Section 1, parse parameters
    mydata = CORD(None)
    test_dataset = mydata.get_data(split='test')
    print(test_dataset)
    doc1 = test_dataset[0]
    print(doc1['input_ids'])

    '''
    Dataset({
        features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
        num_rows: 100
    })
    '''

