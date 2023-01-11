from tqdm import tqdm
import json
import os
from torch_geometric.data import Data
from PIL import Image
import numpy as np
import torch
import sys
sys.path.append('../')
from utils import util, embedding
import math
from os import path
import collections
import ast

class CORD4G:

    def __init__(self, opt=None):
        self.opt = opt
        self.embed = embedding.Embedding(opt)
        self.train_graphs, self.train_node_labels, self.train_edge_labels, self.train_features, self.label2idx = self.graph_generate('train')
        self.test_graphs, self.test_node_labels, self.test_edge_labels, self.test_features, _ = self.graph_generate('test', self.label2idx)
        opt.num_labels = len(self.label2idx)
        # if opt.task_type == 'link-inary':
        #     self.balance_edges()
        print('---train---')
        self.print_statis(self.train_node_labels,'train_node')
        self.print_statis(self.train_edge_labels,'train_edge')
        print('---test---')
        self.print_statis(self.test_node_labels,'test_node')
        self.print_statis(self.test_edge_labels,'test_edge')

    def print_statis(self, labels, split='train'):
        all_labels = []
        for nl in labels: all_labels += nl
        print(split+ ' num:', len(all_labels))
        print(split + ' distri:', collections.Counter(all_labels))

    # def balance_edges(self):
    #     for graph in self.train_graphs+self.test_graphs:
    #         ids = []
    #         u, v = graph.edge_index 
    #         labels = graph.y
    #         counterpart =False
    #         for i in range(0,len(labels)):
    #             if labels[i]>0: 
    #                 ids.append(i)
    #                 counterpart=True
    #             else:
    #                 if counterpart==True:
    #                     ids.append(i)
    #                     counterpart = False
    #         # regenerate 
    #         graph.y = graph.y[ids]
    #         graph.edge_attr = graph.edge_attr[ids]

    #         edge_index = [u[ids].numpy(),v[ids].numpy()]
    #         graph.edge_index = torch.tensor(edge_index)

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
    def _quad_to_box(self, quad):
    # test 87 is wrongly annotated
        box = (
            max(0, quad["x1"]),
            max(0, quad["y1"]),
            quad["x3"],
            quad["y3"]
        )
        if box[3] < box[1]:
            bbox = list(box)
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
            box = tuple(bbox)
        if box[2] < box[0]:
            bbox = list(box)
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
            box = tuple(bbox)
        return box

    # graph generation
    def graph_generate(self,split='test', label2idx = None):
        if split == 'train':
            data_path = self.opt.cord_train
        elif split == 'test':
            data_path = self.opt.cord_test
        else:
            print('split wrong:', split)
            return  
        # 1 return pt if already exists
        if path.exists(data_path+split+'.pt'):
            return torch.load(data_path+split+'.pt')

        graph_list, node_labels, edge_labels, features = self.graph4CORD(data_path = data_path)
        # 2.1 encode/get label list
        if not label2idx:
            label_set = set()
            for item in graph_list:
                y_nrole = item['y_nrole']
                for label in y_nrole: 
                    label_set.add(label)
            label2idx = {l:i for i,l in enumerate(sorted(list(label_set)))}

        # 2.2 encode the graphs
        graphs = []
        for item in graph_list:
            graph = Data(x = torch.tensor(item['x'],dtype=torch.float), 
                    edge_index=torch.tensor(item['edge_index'],dtype=torch.long), 
                    edge_attr= torch.tensor(item['edge_attr'],dtype=torch.long), 
                    y_dist = torch.tensor(item['y_dist'], dtype=torch.float),
                    y_direct = torch.tensor(item['y_direct'],dtype=torch.long),   #
                    y_nrole = torch.tensor([label2idx[l] for l in item['y_nrole']], dtype=torch.long),
                    seg_id = torch.tensor(item['seg_id'], dtype=torch.long),
                    doc_id = torch.tensor(item['doc_id'], dtype=torch.long)
            )
            graphs.append(graph)
        
        # 3 save and return
        torch.save([graphs, node_labels, edge_labels, features, label2idx], data_path+split+'.pt')
        return graphs, node_labels, edge_labels, features,label2idx


    def graph4CORD(self,data_path=None):
        '''
        data_path (str): path to where the data is tored
        rtype (list): graphs, nodes and edge labels, features
        '''
        # all graph level
        graph_list, node_labels, edge_labels = [],[],[]
        features = {'paths':[], 'texts':[], 'boxs':[]}

        # each graph level
        # for json_file in tqdm(os.listdir(os.path.join(data_path,'adjusted_annotations')), desc='initializing the graphs'):
        ann_dir = os.path.join(data_path, "json")
        for doc_idx, json_file in enumerate(sorted(os.listdir(ann_dir))):
            img_name = f'{json_file.split(".")[0]}.png'
            img_path = os.path.join(data_path,'image',img_name)
            image,size = self._load_image(img_path)
            features['paths'].append(img_path)

            #!! graph-level info collection !!
            boxs,texts,seg_ids,nl = [],[],[],[] # raw node label (nl)
            pair_labels = []
            pair2label = {}

            # my adding
            sizes = []

            file_path = os.path.join(ann_dir, json_file)
            with open(file_path, 'r',encoding='utf8') as f:
                data = json.load(f)

            seg_id = 0
            # iterate node-level
            for seg in data['valid_line']:
                cur_line_bboxes = []
                seg_words = []
                words, label = seg["words"], seg["category"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                
                if label=="other":
                    for w in words:
                        seg_words.append(w['text'])
                        cur_line_bboxes.append(self._normalize_bbox(self._quad_to_box(w["quad"]), size))
                else:
                    seg_words.append(words[0]['text'])
                    cur_line_bboxes.append(self._normalize_bbox(self._quad_to_box(words[0]["quad"]), size))
                    for w in words[1:]:
                        seg_words.append(w['text'])
                        cur_line_bboxes.append(self._normalize_bbox(self._quad_to_box(w["quad"]), size))
                box = self._get_line_bbox(cur_line_bboxes)[0]   # first one
                text = ' '.join(seg_words)
                
                # append values
                boxs.append(box)
                s_width, s_height = boxs[-1][2]-boxs[-1][0], boxs[-1][3] - boxs[-1][1]
                sizes.append([s_width,s_height])
                texts.append(text)
                nl.append(label)
                seg_ids.append(seg_id)  # seg_id

                seg_id += 1
            # graph-level save
            node_labels.append(nl)  # header, key, value, others
            features['texts'].append(texts)
            features['boxs'].append(boxs)

            # generate additional features: edges between nodes
            if self.opt.g_neib == '8direct':
                edge_index, edge_attr = util.rolling_neibor_matrix(Image.open(img_path).size, boxs)
                y_direct = [direct for _,direct in edge_attr]
            elif self.opt.g_neib == 'knn':      
                edge_index, edge_attr = util.KNN(Image.open(img_path).size, boxs,10)
                y_direct = [angle//45 for _,angle in edge_attr]
            u, v = edge_index    # add, [2 * num_edge], [num_edge * 2]
            # y_dist = [round(dist,2) for dist,_ in edge_attr]
            y_dist = [round(math.log(dist+1),3) for dist,_ in edge_attr]  # project to [0-7]
            

            doc_ids = [doc_idx for _ in range(len(seg_ids))]

            # creating the single graph
            # initialize the vector: s_i, shape(302,)
            x = []
            for i,text in enumerate(texts):
                x.append(np.concatenate((self.embed.get_text_vect(text),np.array(sizes[i])), axis=-1))
            # graph
            graph = {'x': x, 
                    'edge_index': edge_index, 
                    'edge_attr': edge_attr, 
                    'y_dist' : y_dist,
                    'y_direct' : y_direct,   #
                    'y_nrole' : nl,
                    'seg_id' : seg_ids,
                    'doc_id' : doc_ids
            }
            graph_list.append(graph)
        return graph_list, node_labels, edge_labels, features



if __name__ == '__main__':
    all_distances = []

    data_path = '../../data/FUNSD/testing_data/'

    data = MyDataset()

    graphs, node_labels, edge_labels, features = data._fromFUNSD(data_path=data_path)
    print(len(graphs))

    # train: 149, test: 50
    # import seaborn as sns
    # sns.displot(data=dists, kde=True)


