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

class FUNSD:

    def __init__(self, opt=None):
        self.opt = opt
        self.embed = embedding.Embedding(opt)
        self.train_graphs, self.train_node_labels, self.train_edge_labels, self.train_features = self.graph4FUNSD('train')
        self.test_graphs, self.test_node_labels, self.test_edge_labels, self.test_features = self.graph4FUNSD('test')
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


    def graph4FUNSD(self,split='test'):
        '''
        data_path (str): path to where the data is tored
        rtype (list): graphs, nodes and edge labels, features
        '''

        if split == 'train':
            data_path = self.opt.funsd_train
        elif split == 'test':
            data_path = self.opt.funsd_test
        else:
            print('split wrong:', split)
            return
        
        # return pt if already exists
        if path.exists(data_path+split+'.pt'):
            return torch.load(data_path+split+'.pt')
        
        nlabel_dict = {'question':0,'answer':1, 'header':2, 'other':3}

        # all graph level
        graphs, node_labels, edge_labels = [],[],[]
        features = {'paths':[], 'texts':[], 'boxs':[]}


        glob_seg_id = 0
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
                # ids.append(seg['id'])
                ids.append(glob_seg_id)
                glob_seg_id+=1
                
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
            edge_index, edge_attr = util.KNN(Image.open(img_path).size, boxs, k=10)
            u, v = edge_index    # add, [2 * num_edge], [num_edge * 2]
            y_dist = [round(math.log(dist+1),2) for dist,_ in edge_attr]  # project to [0-7]
            y_direct = [angle//45 for _,angle in edge_attr]  # transforom into 8 directions

            # edge labels (binary or)
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
                x.append(np.concatenate((self.embed.get_text_vect(text),np.array(sizes[i])), axis=-1))
            # graph
            graph = Data(x = torch.tensor(x,dtype=torch.float), 
                    edge_index=torch.tensor(edge_index,dtype=torch.long), 
                    edge_attr= torch.tensor(edge_attr,dtype=torch.long), 
                    y = torch.tensor(el,dtype=torch.float),  # is edge link label
                    y_dist = torch.tensor(y_dist, dtype=torch.float),
                    y_direct = torch.tensor(y_direct,dtype=torch.long),   #
                    y_nrole = torch.tensor(y_nrole, dtype=torch.long),
                    seg_id = torch.tensor(ids, dtype=torch.long)
                )
            graphs.append(graph)

        torch.save([graphs, node_labels, edge_labels, features], data_path+split+'.pt')
        return graphs, node_labels, edge_labels, features



if __name__ == '__main__':
    all_distances = []

    data_path = '../../data/FUNSD/testing_data/'

    data = MyDataset()

    graphs, node_labels, edge_labels, features = data._fromFUNSD(data_path=data_path)
    print(len(graphs))

    # train: 149, test: 50
    # import seaborn as sns
    # sns.displot(data=dists, kde=True)


