from tqdm import tqdm
import json
import os
from torch_geometric.data import Data


def fully_connected(ids):
    u,v = [],[]

    for id in ids:
        u.extend([id for i in range(len(ids)) if i!=id])
        v.extend([i for i in range(len(ids)) if i!=id])
    return u,v


def _fromFUNSD(data_path): 
    '''
    data_path (str): path to where the data is tored
    rtype (list): graphs, nodes and edge labels, features
    '''

    # all graph level
    graphs, node_labels, edge_labels = [],[],[]
    features = {'paths':[], 'texts':[], 'boxs':[]}
    
    # each graph level
    for json_file in tqdm(os.listdir(os.path.join(data_path,'adjusted_annotations')), desc='initializing the graphs'):
        img_name = f'{json_file.split(".")[0]}.jpg'
        img_path = os.path.join(data_path,'images',img_name)
        features['paths'].append(img_path)

        #!! graph-level info collection !!
        boxs,texts,ids,nl = [],[],[],[] # raw node label (nl)
        pair_labels = []
        id2label = {}
        pair2label = {}

        with open(os.path.join(data_path,'adjusted_annotations', json_file), 'r') as f:
            json_form = json.load(f)['form']

        # node-level
        for seg in json_form:
            boxs.append(seg['box'])
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

        # getting edges
        # if self.edge_type == 'fully':
        #     u,v = self.fully_connected()
        # elif self.edge_type == 'knn':
        #     u,v = self._knn(Image.open(img_path).size, boxs)
        # else:
        #     raise Exception('pls choose correct edge types')
        u,v = fully_connected(ids)

        
        # edge labels (binary or )
        el = []
        for e in zip(u,v):
            edge = [e[0], e[1]]
            if edge in pair_labels: el.append('pair')
            else: el.append('none')
        edge_labels.append(el)

        # creating the graph
        graph = Data(x = ids, edge_index=pair_labels, y = el)
        graphs.append(graph)
    return graphs, node_labels, edge_labels, features


data_path = '../../data/FUNSD/training_data/'

graphs, node_labels, edge_labels, features = _fromFUNSD(data_path=data_path)
print(len(graphs))

# train: 149, test: 50
