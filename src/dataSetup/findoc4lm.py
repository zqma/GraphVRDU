from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoProcessor



class FinDoc:
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

        #
        # print(self.test_dataset)
        # doc1 = self.test_dataset[0]
        # print(doc1['input_ids'])

    def load_dataaset_from_json(self,split='train'):
        if split == 'train':
            data_path = self.opt.funsd_train
        elif split == 'test':
            data_path = self.opt.funsd_test
        else:
            print('split wrong:', split)
            return
        
        # return pt if already exists
        # if path.exists(data_path+split+'.pt'):
        #     return torch.load(data_path+split+'.pt')
        
        nlabel_dict = {'question':0,'answer':1, 'header':2, 'other':3}

        # all graph level
        # graphs, node_labels, edge_labels = [],[],[]
        # features = {'paths':[], 'texts':[], 'boxs':[]}

        my_dict = {
            'id':[],
            'texts':[],
            'bboxes':[],
            'labels':[]
            'image':[]
        }
        
        # each graph level
        for json_file in tqdm(os.listdir(os.path.join(data_path,'adjusted_annotations')), desc='initializing the graphs'):
            img_name = f'{json_file.split(".")[0]}.png'
            img_path = os.path.join(data_path,'images',img_name)
            features['paths'].append(img_path)

            #!! graph-level info collection !!
            boxs,texts,ids,nl = [],[],[],[] # raw node label (nl)
            pair_labels = []
            id2label = {}
            pair2label = {}

            # my adding
            sizes = []

            with open(os.path.join(data_path,'adjusted_annotations', json_file), 'r') as f:
                json_form = json.load(f)['form']

            # node-level
            for seg in json_form:
                boxs.append(seg['box'])
                s_width, s_height = boxs[-1][2]-boxs[-1][0], boxs[-1][3] - boxs[-1][1]
                sizes.append([s_width,s_height])
                texts.append(seg['text'])
                nl.append(seg['label'])
                ids.append(seg['id'])
                id2label[seg['id']] = seg['label']
                for pair in seg['linking']:
                    pair_labels.append(pair)    # is it directed??
            # To re-index the matrix by ordered idx
            for i,pair in enumerate(pair_labels):
                pair_label = id2label[pair[0]] + '_' + id2label[pair[1]]
                if pair_label != 'question_answer': continue    # filter non-QA values
                # re-index the edge matrix w.r.t the x
                pair_labels[i] = [ids.index(pair[0]), ids.index(pair[1])]
                pair2label[i] = pair_label

            # graph-level save
            node_labels.append(nl)  # header, key, value, others
            features['texts'].append(texts)
            features['boxs'].append(boxs)

            y_nrole = [nlabel_dict[label] for label in nl]

            # getting edges
            # if self.edge_type == 'fully':
            #     u,v = self.fully_connected()
            # elif self.edge_type == 'knn':
            #     u,v = self._knn(Image.open(img_path).size, boxs)
            # else:
            #     raise Exception('pls choose correct edge types')
            # u,v = fully_connected(ids)
            edge_index, edge_attr = util.KNN(Image.open(img_path).size, boxs, k=6)
            u, v = edge_index    # add, [2 * num_edge], [num_edge * 2]
            y_dist = [round(math.log(dist+1),2) for dist,_ in edge_attr]  # project to [0-7]
            y_direct = [angle//45 for _,angle in edge_attr]  # transforom into 8 directions

            # edge labels (binary or )
            el = []
            for e in zip(u,v):
                edge = [e[0], e[1]]
                if edge in pair_labels: 
                    el.append(1)    # is link (e.g., question answering)
                else: 
                    el.append(0)    # not link (e.g., neibor but no QA relation)
            edge_labels.append(el)

            # creating the single graph
            # initialize the vector: s_i, shape(302,)
            x = []
            for i,text in enumerate(texts):
                x.append(np.concatenate((embedding.get_text_vect(text),np.array(sizes[i])), axis=-1))
            # graph
            graph = Data(x = torch.tensor(x, dtype=torch.float), 
                    edge_index=torch.tensor(edge_index,dtype=torch.long), 
                    edge_attr= torch.tensor(edge_attr,dtype=torch.long), 
                    y = torch.tensor(el,dtype=torch.float),  # is edge link label
                    y_dist = torch.tensor(y_dist, dtype=torch.float),
                    y_direct = torch.tensor(y_direct,dtype=torch.long),   #
                    y_nrole = torch.tensor(y_nrole, dtype=torch.long)
                )
            print(graph)
            graphs.append(graph)

        torch.save([graphs, node_labels, edge_labels, features], data_path+split+'.pt')
        return graphs, node_labels, edge_labels, features



    def prepare_one_doc(self,doc):
        images = doc[self.image_col_name] ##if you use an image path, this will need to be updated to read the image in
        words = doc[self.text_col_name]
        boxes = doc[self.boxes_col_name]
        word_labels = doc[self.label_col_name]
        encoding = self.processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length") # must put return tensor
        return encoding

    # def get_data(self,split=''):
    #     for (pdf,page_idx),grp in df.groupby(['pdf','page_index']): # rank by docs and pages
    #         # img = Image.open(page_idx)




    def get_data(self,split='train',shuffle=True):
        # return self.prepare_one_doc(self.dataset[split])
        samples = self.dataset[split]   # get split
        trainable_samples = samples.map(
            self.prepare_one_doc,
            batched=True,
            remove_columns=samples.column_names,    # remove original features
            # features = Features({
            #     'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            #     'input_ids': Sequence(feature=Value(dtype='int64')),
            #     'attention_mask': Sequence(Value(dtype='int64')),
            #     'bbox': Array2D(dtype="int64", shape=(512, 4)),
            #     'labels': Sequence(feature=Value(dtype='int64')),}
            # )
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

    def open_pic(img_path):    
        image = Image.open("/83443897.png")
        image = image.convert("RGB")
        return image


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

