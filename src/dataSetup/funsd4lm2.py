from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoProcessor
from datasets import Dataset,DatasetDict
import os,json
from PIL import Image


class FUNSD:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        DatasetDict({
            train: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 149?
            })
            validation and test 
            })
        })
        '''
        self.image_col_name = "image"
        self.text_col_name = "tokens"
        self.boxes_col_name = "bboxes"
        self.label_col_name = "ner_tags"
        self.seg_col_name = 'seg_ids'
        # load data
        # self.dataset = load_dataset("nielsr/funsd-layoutlmv3")
        self.dataset = self.get_train_test()
        print('--dataset:--',self.dataset)

        opt.id2label, opt.label2id, opt.label_list = self._get_label_map(self.dataset)
        opt.num_labels = len(opt.label_list)
        
        # encode label class (target)
        self.dataset['train'] = self.encode_class(self.dataset['train'])
        self.dataset['test'] = self.encode_class(self.dataset['test'])

        # prepare for getting trainable data
        # 6 labels: {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir, apply_ocr=False)    # wrap of featureExtract & tokenizer
        
        # process to: 'input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values']
        self.train_dataset, self.test_dataset = self.get_data('train'), self.get_data('test')


    def _prepare_one_doc(self,doc):
        doc_id = doc['id']
        seg_ids = doc['seg_ids']
        images = doc[self.image_col_name] ##if you use an image path, this will need to be updated to read the image in
        words = doc[self.text_col_name]
        boxes = doc[self.boxes_col_name]
        word_labels = doc[self.label_col_name]
        encoding = self.processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length") # must put return tensor
        # encoding['gvect'] = self.get_graph_vects(doc[self.seg_col_name])

        return encoding

    def get_data(self,split='train',shuffle=True):
        # return self.prepare_one_doc(self.dataset[split])
        samples = self.dataset[split]   # get split
        trainable_samples = samples.map(
            self._prepare_one_doc,  # process new features
            batched=True,   # 
            remove_columns=samples.column_names,    # remove old features
            features = Features({   # specify new feature types
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Sequence(feature=Value(dtype='int64')),
                # 'labels': ClassLabel(num_classes=7,names=['O','B-HEADER','I-HEADER','B-QUESTION','I-QUESTION','B-ANSWER', 'I-ANSWER']),
            })
        ).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        return trainable_samples


    def _get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def _get_label_map(self,dataset,split='train'):
        features = dataset[split].features
        column_names = dataset[split].column_names

        if isinstance(features[self.label_col_name].feature, ClassLabel):
            label_list = features[self.label_col_name].feature.names
            # No need to convert the labels since they are already ints.
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        else:
            label_list = self._get_label_list(dataset[split][self.label_col_name])
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        return id2label, label2id, label_list


    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)
    def _normalize_bbox(self,bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]
    def _get_line_bbox(self,bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox


    # one doc per img (not multiple pages)
    def load_samples(self, base_dir):
        # logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(base_dir, "adjusted_annotations")
        img_dir = os.path.join(base_dir, "images")
        for doc_idx, file in enumerate(sorted(os.listdir(ann_dir))):
            # print('---doc id:---',doc_idx)
            tokens = []
            bboxes = []
            ner_tags = []
            seg_ids = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = self._load_image(image_path)
            seg_id = 0
            for item in data["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                text = item['text']
                if text.strip()=='': continue
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        seg_ids.append(seg_id)
                        ner_tags.append("O")
                        cur_line_bboxes.append(self._normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    seg_ids.append(seg_id)
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(self._normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        seg_ids.append(seg_id)
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(self._normalize_bbox(w["box"], size))
                cur_line_bboxes = self._get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
                seg_id +=1
            yield {"id": doc_idx, "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                        "image": image,
                        "seg_ids":seg_ids}


    def get_train_test(self):
        ann_dir = os.path.join(self.opt.funsd_train, "adjusted_annotations")
        files = os.listdir(ann_dir)
        print('====',len(files))
        train = Dataset.from_generator(self.load_samples, gen_kwargs={'base_dir':self.opt.funsd_train})
        test = Dataset.from_generator(self.load_samples, gen_kwargs={'base_dir':self.opt.funsd_test})
        return DatasetDict({
            "train" : train , 
            "test" : test 
        })


    def encode_class(self,dataset):
        dst_feat = ClassLabel(names = self.opt.label_list)
        # Mapping Labels to IDs
        def map_label2id(example):
            example[self.label_col_name] = [dst_feat.str2int(ner_label) for ner_label in example[self.label_col_name]]
            return example
        dataset = dataset.map(map_label2id, batched=True)
        # type to ClassLabel object
        # dataset = dataset.cast_column(self.label_col_name, dst_feat)
        return dataset


if __name__ == '__main__':
    # Section 1, parse parameters
    mydata = FUNSD(None)
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

