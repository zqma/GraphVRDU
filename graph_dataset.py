from spektral.data import Dataset
from spektral.data import Graph
from spektral.datasets import TUDataset
from scipy.sparse import csr_array
import h5py
import numpy as np

#the dataset of graph object, with x:node, a:adjacency, e:edge, y:class
class DocDataset(Dataset):
    """
    A dataset of document graph
    """
    def __init__(self, local_data, use_feature="text", return_edge=False, transforms=None, return_sparse_a=True, return_sparse_e=True, edge_abs=False,**kwargs):
        self.local_data = local_data
        self.use_feature = use_feature
        self.return_edge = return_edge
        self.return_sparse_a = return_sparse_a
        self.return_sparse_e = return_sparse_e
        self.edge_abs = edge_abs
        self.transforms = transforms
        self.m = []
        super().__init__(**kwargs)
        
    def read(self,):
        output = []
        
        with h5py.File(self.local_data, "r") as f:
            a_group_key = list(f.keys())[0]
            grps = list(f[a_group_key])
            i = 0
            for grp in grps:
                sample = f[a_group_key][grp]
                
                graph = self.convert(sample)
                
                
                if self.transforms:
                    adjacency = self.transforms(graph)
                output.append(graph)
                
               
            
                
        #print(graph.x.shape, graph.e.shape)        
        return output
                
                
    def convert(self, group, n_label= 16 ):
        #load embeddings for building graph
        text_emb = group["text_emb"][()]
        img_emb = group["img_emb"][()]
        coors = group["coordinates"][()]
        label = group["label"][()]
        
        d = text_emb.shape[-1]
        text_emb = np.reshape(text_emb, (-1, d))
        img_emb = np.reshape(img_emb, (-1, d))
        
        if self.use_feature == "text":
            node_emb = text_emb
            
        elif feature == "img":
            node_emb = img_emb
        else:
            node_emb = np.concatenate((text_emb, img_emb), axis=1)
        
        #make achor for representing the doc   
        anchor = np.zeros((1, node_emb.shape[1]))
        anchor_coor = np.zeros((1, coors.shape[1]))
        
        node_emb = np.concatenate((anchor, node_emb), axis=0)
        
        n_node = node_emb.shape[0]
        adjacency = np.ones((n_node, n_node)) - np.identity(n_node)
        
        
        edge_emb = None
        if self.return_edge:
            coors = np.concatenate((anchor_coor, coors), axis=0)
            edge_emb = np.zeros((n_node, n_node, coors.shape[1]))
        
        
            #compute edge feature
            if self.edge_abs:
                edge_emb = abs(coors - coors[:,np.newaxis])+0.01
            
                #normalize edge to the range(0,1), note that large distance -> smaller weight
    
                m = np.max(edge_emb, axis=(0, 1))
        
                beta = 0.01
                edge_emb = np.cos(edge_emb*np.pi / (2*m[np.newaxis, :])-beta*np.sign(edge_emb))
            else:
                edge_emb = abs(coors - coors[:,np.newaxis])
                #m = np.max(edge_emb, axis=(0, 1))
                #edge_emb = edge_emb/(m+0.01)
                edge_emb = edge_emb.astype(int)
                
            
        #convert dense adjacency matrice to sparse format
        if self.return_sparse_a:
            non_zero_a = np.nonzero(adjacency) 
            adjacency = csr_array((adjacency[non_zero_a], non_zero_a))
        
        #convert dense edge matrice to sparse format 
        if self.return_sparse_e:
            non_dig_e = np.where(~np.eye(edge_emb.shape[0],dtype=bool))
            edge_emb = edge_emb[non_dig_e]
            
        
        
        #convert label to one-hot encoding
        res = np.zeros(n_label)
        res[label] = 1
        label = res
        return Graph(a=adjacency , x=node_emb , e=edge_emb , y= label)

if __name__ == "__main__":
    path = "graph_ele.hdf5"
    dataset = DocDataset(path)
    #dataset = TUDataset('PROTEINS')
    print(dataset)
    print(type(dataset[1].a))