from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoProcessor



class CORD:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        DatasetDict({
            train: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 800
            })
            validation: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 100
            })
            test: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 100
            })
        })
        '''
        self.image_col_name = "image"
        self.text_col_name = "words"
        self.boxes_col_name = "bboxes"
        self.label_col_name = "ner_tags"
        # load data
        self.dataset = load_dataset("nielsr/cord-layoutlmv3")


        # prepare for getting trainable data
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir, apply_ocr=False)    # wrap of featureExtract & tokenizer
        opt.id2label, opt.label2id, opt.label_list = self.get_label_map(self.dataset)
        opt.num_labels = len(opt.label_list)
        self.train_dataset, self.test_dataset = self.get_data('validation'), self.get_data('test')
        '''
        {0: 'O', 1: 'B-MENU.NM', 2: 'B-MENU.NUM', 3: 'B-MENU.UNITPRICE', 4: 'B-MENU.CNT', 5: 'B-MENU.DISCOUNTPRICE', 6: 'B-MENU.PRICE', 7: 'B-MENU.ITEMSUBTOTAL', 8: 'B-MENU.VATYN', 9: 'B-MENU.ETC', 10: 'B-MENU.SUB_NM', 11: 'B-MENU.SUB_UNITPRICE', 12: 'B-MENU.SUB_CNT', 13: 'B-MENU.SUB_PRICE', 14: 'B-MENU.SUB_ETC', 15: 'B-VOID_MENU.NM', 16: 'B-VOID_MENU.PRICE', 17: 'B-SUB_TOTAL.SUBTOTAL_PRICE', 18: 'B-SUB_TOTAL.DISCOUNT_PRICE', 19: 'B-SUB_TOTAL.SERVICE_PRICE', 20: 'B-SUB_TOTAL.OTHERSVC_PRICE', 21: 'B-SUB_TOTAL.TAX_PRICE', 22: 'B-SUB_TOTAL.ETC', 23: 'B-TOTAL.TOTAL_PRICE', 24: 'B-TOTAL.TOTAL_ETC', 25: 'B-TOTAL.CASHPRICE', 26: 'B-TOTAL.CHANGEPRICE', 27: 'B-TOTAL.CREDITCARDPRICE', 28: 'B-TOTAL.EMONEYPRICE', 29: 'B-TOTAL.MENUTYPE_CNT', 30: 'B-TOTAL.MENUQTY_CNT', 31: 'I-MENU.NM', 32: 'I-MENU.NUM', 33: 'I-MENU.UNITPRICE', 34: 'I-MENU.CNT', 35: 'I-MENU.DISCOUNTPRICE', 36: 'I-MENU.PRICE', 37: 'I-MENU.ITEMSUBTOTAL', 38: 'I-MENU.VATYN', 39: 'I-MENU.ETC', 40: 'I-MENU.SUB_NM', 41: 'I-MENU.SUB_UNITPRICE', 42: 'I-MENU.SUB_CNT', 43: 'I-MENU.SUB_PRICE', 44: 'I-MENU.SUB_ETC', 45: 'I-VOID_MENU.NM', 46: 'I-VOID_MENU.PRICE', 47: 'I-SUB_TOTAL.SUBTOTAL_PRICE', 48: 'I-SUB_TOTAL.DISCOUNT_PRICE', 49: 'I-SUB_TOTAL.SERVICE_PRICE', 50: 'I-SUB_TOTAL.OTHERSVC_PRICE', 51: 'I-SUB_TOTAL.TAX_PRICE', 52: 'I-SUB_TOTAL.ETC', 53: 'I-TOTAL.TOTAL_PRICE', 54: 'I-TOTAL.TOTAL_ETC', 55: 'I-TOTAL.CASHPRICE', 56: 'I-TOTAL.CHANGEPRICE', 57: 'I-TOTAL.CREDITCARDPRICE', 58: 'I-TOTAL.EMONEYPRICE', 59: 'I-TOTAL.MENUTYPE_CNT', 60: 'I-TOTAL.MENUQTY_CNT'}
        '''

    def prepare_one_doc(self,doc):
        images = doc[self.image_col_name] ##if you use an image path, this will need to be updated to read the image in
        words = doc[self.text_col_name]
        boxes = doc[self.boxes_col_name]
        word_labels = doc[self.label_col_name]
        encoding = self.processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length") # must put return tensor
        return encoding

    def get_data(self,split='train',shuffle=True):
        # return self.prepare_one_doc(self.dataset[split])
        samples = self.dataset[split]   # get split

        trainable_samples = samples.map(
            self.prepare_one_doc,
            batched=True,
            remove_columns=samples.column_names,    # remove original features
            features = Features({
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Sequence(feature=Value(dtype='int64')),}
            )
        ).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        return trainable_samples


    def get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def get_label_map(self,dataset,split='train'):
        features = dataset[split].features
        column_names = dataset[split].column_names
        

        if isinstance(features[self.label_col_name].feature, ClassLabel):
            label_list = features[self.label_col_name].feature.names
            # No need to convert the labels since they are already ints.
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        else:
            label_list = self.get_label_list(dataset[split][self.label_col_name])
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        return id2label, label2id, label_list
    



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

