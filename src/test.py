from transformers import LayoutLMv2Processor
from transformers import LayoutLMv2FeatureExtractor
from PIL import Image
import pytesseract

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
feature_extractor = LayoutLMv2FeatureExtractor(tesseract_config='--psm 1 --oem 1')

image_path = "/home/ubuntu/python_projects/GraphVRDU/data/FUNSD/testing_data/images/82491256.png"
data = pytesseract.image_to_data(Image.open(image_path),output_type='dict')
print(type(data))
print(len(data['text']),data['text']) # 

image = Image.open(
    image_path
).convert("RGB")

# 
encoding = processor(
    image, return_tensors="pt",return_token_type_ids=True 
)  # you can also add all tokenizer parameters here such as padding, truncation
input_ids = encoding['input_ids']
print(input_ids)

word_ids = encoding['word_ids']
print(word_ids)
# tokens = processor.decode(input_ids[0])
# print(tokens)
# for i,token in enumerate(tokens): # 520 length
#     print(i,token)

encoded_inputs = feature_extractor(image)
for k in encoded_inputs.keys: print(k)
# examples['image'] = encoded_inputs.pixel_values
words = encoded_inputs.words
boxes = encoded_inputs.boxes
print(len(words[0]),words)
print(boxes)


# question = "What's his name?"
# encoding = processor(
#     image, question, return_tensors="pt",return_token_type_ids=True 
# )  # you can also add all tokenizer parameters here such as padding, truncation
# for k,v in encoding.items():
#     print(k,':',v)