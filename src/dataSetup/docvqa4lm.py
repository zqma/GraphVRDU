from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoTokenizer
import transformers

import json



class DocVQA:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        Dataset({
            features: ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'start_positions', 'end_positions'],
            num_rows: 50
        })
        '''
        self.dataset = load_dataset(opt.docvqa_dir)
        print(self.dataset)

        # prepare for getting trainable data
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)    #tokenizer
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast) # get sub

        self.train_dataset, self.test_dataset = self.get_data('validation'), self.get_data('test')

        # print(self.test_dataset)
        # doc1 = self.test_dataset[0]
        # print(doc1['input_ids'])


    def get_data(self,split='train',shuffle=True):
        # return self.prepare_one_doc(self.dataset[split])
        samples = self.dataset[split]   # get split
        trainable_samples = samples.map(
            self._prepare_one_batch,
            batched=True, 
            batch_size = self.opt.batch_size,
            remove_columns=samples.column_names,    # remove original features
            features = Features({
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'image': Array3D(dtype="int64", shape=(3, 224, 224)),
                'start_positions': Value(dtype='int64'),
                'end_positions': Value(dtype='int64'),
            })
        ).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        return trainable_samples

    def _prepare_one_batch(self, examples, max_length=512):
        # take a batch 
        questions = examples['question']
        words = examples['words']
        boxes = examples['boxes']

        # encode it
        encoding = self.tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

        # # next, add start_positions and end_positions
        # start_positions = []
        # end_positions = []
        # answers = examples['answers']
        # # for every example in the batch:
        # for batch_index in range(len(answers)):
        #     print("Batch index:", batch_index)
        #     cls_index = encoding.input_ids[batch_index].index(self.tokenizer.cls_token_id)
        #     # try to find one of the answers in the context, return first match
        #     words_example = [word.lower() for word in words[batch_index]]
        #     for answer in answers[batch_index]:
        #         match, word_idx_start, word_idx_end = self.subfinder(words_example, answer.lower().split())
        #         if match:
        #             break
        #         # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
        #         if not match:
        #             for answer in answers[batch_index]:
        #                 for i in range(len(answer)):
        #                 # drop the ith character from the answer
        #                 answer_i = answer[:i] + answer[i+1:]
        #                 # check if we can find this one in the context
        #                 match, word_idx_start, word_idx_end = self.subfinder(words_example, answer_i.lower().split())
        #                 if match:
        #                     break
        #         # END OF EXPERIMENT 
            
        #     if match:
        #     sequence_ids = encoding.sequence_ids(batch_index)
        #     # Start token index of the current span in the text.
        #     token_start_index = 0
        #     while sequence_ids[token_start_index] != 1:
        #         token_start_index += 1

        #     # End token index of the current span in the text.
        #     token_end_index = len(encoding.input_ids[batch_index]) - 1
        #     while sequence_ids[token_end_index] != 1:
        #         token_end_index -= 1
            
        #     word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]
        #     for id in word_ids:
        #         if id == word_idx_start:
        #         start_positions.append(token_start_index)
        #         break
        #         else:
        #         token_start_index += 1

        #     for id in word_ids[::-1]:
        #         if id == word_idx_end:
        #         end_positions.append(token_end_index)
        #         break
        #         else:
        #         token_end_index -= 1
            
        #     print("Verifying start position and end position:")
        #     print("True answer:", answer)
        #     start_position = start_positions[batch_index]
        #     end_position = end_positions[batch_index]
        #     reconstructed_answer = tokenizer.decode(encoding.input_ids[batch_index][start_position:end_position+1])
        #     print("Reconstructed answer:", reconstructed_answer)
        #     print("-----------")
            
        #     else:
        #     print("Answer not found in context")
        #     print("-----------")
        #     start_positions.append(cls_index)
        #     end_positions.append(cls_index)
        
        # encoding['image'] = examples['image']
        # encoding['start_positions'] = start_positions
        # encoding['end_positions'] = end_positions

        # return encoding


    def subfinder(self,words_list, answer_list):  
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

    def load_json(self,json_file):
        with open(json_file) as f:
            data = json.load(f)
            # data.keys() -> dataset_name, dataset_version, dataset_split, data
    
    def load_QAs(self,json_file, is_training):
        with open(json_file) as f:
            data = json.load(f)
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        




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

