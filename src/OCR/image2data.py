import json
import pandas as pd
from transformers import AutoTokenizer
import transformers

# find the index, of the answer for raw tokens;
def _subfinder(words_list, answer_list):  
    print('input words:',words_list)
    print('input ans:',answer_list)
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


tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/resources/layoutlmv3.funsd')
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)



# tokenizer needed to be delivered in; 
def token_idx_for_question_answering(question, words, boxes):
  encoding = tokenizer(question, words,boxes=boxes )

  sequence_ids = encoding.sequence_ids()
  # 1 positions for A part in the whole sequence positions
  # 1.1 Start token index of the current span in the text a (answer part).
  token_start_index = 0
  while sequence_ids[token_start_index] != 1:
      token_start_index += 1

  # 1.2 End token index of the current span in the text a (answer part).
  token_end_index = len(encoding.input_ids) - 1
  while sequence_ids[token_end_index] != 1:
      token_end_index -= 1
  text = tokenizer.decode(encoding.input_ids[token_start_index:token_end_index+1])

  # 2. positions for A-answer in the whole sequence
  # 2.1. get the A word ids 
  word_ids = encoding.word_ids()
  # e.g., [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10]

  s,e = token_start_index,token_end_index
  
  return word_ids,(s,e),text

# tokenizer needed to be delivered in; 
def token_idx_for_token_classification(words, boxes):
  encoding = tokenizer(text = words,boxes=boxes)  # input_ids, attention_mask, bbox, (hidden returns: word_ids, sequence_ids)
  print('boxes:',encoding['bbox'])

  sequence_ids = encoding.sequence_ids()
  # 1 positions for A part in the whole sequence positions
  # 1.1 Start token index of the current span in the text a (answer part).
  token_start_index = 0
  while sequence_ids[token_start_index] != 0:
      token_start_index += 1

  # 1.2 End token index of the current span in the text a (answer part).
  token_end_index = len(encoding.input_ids) - 1
  while sequence_ids[token_end_index] != 0:
      token_end_index -= 1
  text = tokenizer.decode(encoding.input_ids[token_start_index:token_end_index+1])

  # 2. positions for A-answer in the whole sequence
  # 2.1. get the A word ids 
  word_ids = encoding.word_ids()
  # e.g., [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10]

  txt_start,txt_end = token_start_index,token_end_index
  
  return word_ids,(txt_start,txt_end),text



def reconstruct_answer(question,words, boxes, word_idx_start,word_idx_end):
    word_ids,(s,e),text = token_idx_for_question_answering(question, words, boxes)
    text_start,text_end = s,e
    # 2. positions for A-answer in the whole sequence
    # 2.1. get the A word ids 
    sub_word_ids = word_ids[s:e+1]

    token_start_index,token_end_index = s,e

    # 2.2 two pointers, l,r positions, to point to correspond sequence positions; 
    for id in sub_word_ids:
        if id == word_idx_start:
            start_position = token_start_index 
        else:
            token_start_index += 1

    for id in sub_word_ids[::-1]:
        if id == word_idx_end:
            end_position = token_end_index 
        else:
            token_end_index -= 1
    return word_ids, [text_start,text_end], [token_start_index, token_end_index]
    # word_ids (independent), text_part_position_range (in terms of sequence), answer_part_position_range (in terms of sequence), 

# encode the questions and its image words 
# question = "where is it located?"
# words = ["this", "is", "located", "in", "the", "university", "of", "california", "in", "the", "US"]
# boxes = [[1000,1000,1000,1000] for _ in range(len(words))]
# answer = "university of california"

# # token classification detect
# print('===token function===')
# word_ids, text_range, valid_text = token_idx_for_token_classification(words,boxes)
# print(word_ids)
# print(text_range)
# print(valid_text)

# # QA detect
# print('===QA function===')
# match,word_idx_start,word_idx_end = _subfinder(words,answer.split())  # lower case
# print(match, word_idx_start, word_idx_end)
# word_ids, text_range, answer_range = reconstruct_answer(question,words,boxes,word_idx_start,word_idx_end)
# print(word_ids)
# print(text_range)
# print(answer_range)



def _raw_ans_word_idx_range(words, answers):
    # Match trial 1: try to find one of the answers in the context, return first match
    words_example = [word.lower() for word in words]
    for answer in answers:
        match, ans_word_idx_start, ans_word_idx_end = _subfinder(words_example, answer.lower().split())
        if match:
            break
    # Match trial 2: EXPERIMENT (to account for when OCR context and answer don't perfectly match):
    # if not match:
    #     for answer in answers:
    #         for i in range(len(answer)):
    #             # drop the ith character from the answer
    #             answer_i = answer[:i] + answer[i+1:]
    #             # check if we can find this one in the context
    #             match, ans_word_idx_start, ans_word_idx_end = _subfinder(words_example, answer_i.lower().split())
    #             if match:
    #                 break
    # END OF EXPERIMENT 
    return match, ans_word_idx_start, ans_word_idx_end
    
def _ans_index_range(batch_encoding,batch_index, answer_word_idx_start, answer_word_idx_end):
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


def encode_dataset(examples, max_length=512):
    # 1. take a batch 
    questions = examples['question']
    answers = examples['answers']
    words = examples['words']
    boxes = examples['boxes']

    # 2. encode it
    encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)

    # 3. next, add start_positions and end_positions
    ans_start_positions = []
    ans_end_positions = []
    
    # for every example in the batch:
    for batch_index in range(len(answers)):
        print("Batch index:", batch_index)  
        print(questions[batch_index])
        print(answers[batch_index])
        print(words[batch_index])
        # 3.1 step1: match answer range in raw word idx []0,1,2,..] to get e.g., [3,5]
        match, ans_word_idx_start, ans_word_idx_end = _raw_ans_word_idx_range(words[batch_index], answers[batch_index])
        
        # 3.2 step2: match answer range in the sequence e.g., [None, 0,1,2,2,2,3,3,4,5,6,7,7,7, None] to get index range
        if match:
            answer_start_index, answer_end_index = _ans_index_range(encoding,batch_index,ans_word_idx_start, ans_word_idx_end)
            ans_start_positions.append(answer_start_index)
            ans_end_positions.append(answer_end_index)

            print("Verifying start position and end position:===")
            print("True answer:", answer)
            start_position = ans_start_positions[batch_index]
            end_position = ans_end_positions[batch_index]
            reconstructed_answer = tokenizer.decode(encoding.input_ids[batch_index][answer_start_index:answer_end_index+1])
            print("Reconstructed answer:", reconstructed_answer)
            print("-----------")
        
        else:
            cls_index = encoding.input_ids[batch_index].index(tokenizer.cls_token_id)
            print("Answer not found in context")
            print("-----------")
            ans_start_positions.append(cls_index)
            ans_end_positions.append(cls_index)
    # 3.3 append the ans_start, ans_end_index
    encoding['image'] = examples['image']
    encoding['ans_start_positions'] = ans_start_positions
    encoding['ans_end_positions'] = ans_end_positions

    return encoding


question = "where is it located?"
words = ["this", "is", "located", "in", "the", "university", "of", "california", "in", "the", "US"]
boxes = [[1000,1000,1000,1000] for _ in range(len(words))]
answer = ["university of california"]

examples = {}
examples['question'] = [question] * 3
examples['answers'] = [answer] * 3
examples['words'] = [words for _ in range(3)]
examples['boxes'] = [boxes for _ in range(3)]

encode_dataset(examples)


def load_questions(file_path='val'):
    with open(file_path,'rb') as fr:
        # {'dataset_name', 'dataset_version', 'dataset_split', 'data'} 
        data = json.load(fr)
    # questionId, question, image, docId, ucsf_document_id, ucsf_document_page_no, answers, data_split
    # e.g., question: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?
    # e.g., answers: ['TRRF Vice President', 'lee a. waller']

    df = pd.DataFrame(data['data'])
    df.head()   




