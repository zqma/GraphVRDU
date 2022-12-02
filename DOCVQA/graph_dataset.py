from spektral.data import Dataset
from spektral.data import Graph
from spektral.datasets import TUDataset
from scipy.sparse import csr_array
from itertools import combinations
import h5py
import numpy as np

#the dataset of graph object, with x:node, a:adjacency, e:edge, y:class
class DocDataset(Dataset):
    """
    A dataset of document graph
    """
    def __init__(self, local_data, use_feature="text", return_edge=True, transforms=None, return_sparse_a=True, return_sparse_e=True, edge_abs=False,**kwargs):
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
            grps = list(f.keys())
            i = 0
            
            for grp in grps:
                sample = f[grp]
                
                graph = self.convert(sample)
                
                
                if self.transforms:
                    adjacency = self.transforms(graph)
                output.append(graph)
                
                break
            
                
        #print(graph.x.shape, graph.e.shape)        
        return output
    
    def constrain_adj(self, seg_ids):
        
        sub_graph_parents = [0]+ [seg_ids[0][0][0]-1 + 1] + [ids[1]+1 for ids in seg_ids[0]]
        ####append the parent of origin node and question node too
        ####also need to shift the entire list by 1 since we introduced origin
        
        #now we can modify the adjacency matrix, start with connecting parent nodes
        
        parent_con =  list(combinations(sub_graph_parents,2))
        
        connectivity = parent_con 
        #then we can move to the internal connection within subgraphs
        for i in range(1, len(sub_graph_parents)):
             internal_con = list(combinations( [i for i in range(sub_graph_parents[i-1] + 1, sub_graph_parents[i] + 1)], 2))
             connectivity += internal_con
        
        
        #the above connectivity is only one sided, need to be augmented into two-sided
        connectivity_ = [x[::-1] for x in connectivity]
        
        connectivity += connectivity_
        values = np.ones(len(connectivity))
        row, column = zip(*connectivity)
        row = np.array(row)
        column = np.array(column)
    
        return   csr_array((values, (row, column)))  
                
    def convert(self, group, n_label= 16 ):
        #load embeddings for building graph
        text_emb = group["node_embedding"][()]
        
        word_coors = np.squeeze(group["word_coordinate"][()])
        seg_coors = group["seg_coordinate"][()]
        seg_ids = group["seg_ids"][()]
        
        ans_start = group["ans_start"][()]
        ans_end = group["ans_end"][()]
        
        d = text_emb.shape[-1]
        text_emb = np.reshape(text_emb, (-1, d))
       
        
        
        node_emb = text_emb
            
        #make achor for representing the doc   
        anchor = np.zeros((1, node_emb.shape[1]))
        anchor_coor = np.zeros((1, word_coors.shape[1]))
        
        node_emb = np.concatenate((anchor, node_emb), axis=0)
        
        #initialize a adjacency matrix
        n_node = node_emb.shape[0]
        #adjacency = np.zeros((n_node, n_node))  #- np.identity(n_node)
        
        adjacency = self.constrain_adj(seg_ids)
        
        edge_emb = None
        if self.return_edge:
            coors = np.concatenate((anchor_coor, word_coors), axis=0)
            edge_emb = np.zeros((n_node, n_node, word_coors.shape[1]))
        
        
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
        #if self.return_sparse_a:
            #non_zero_a = np.nonzero(adjacency) 
            #adjacency = csr_array((adjacency[non_zero_a], non_zero_a))
        
        #convert dense edge matrice to sparse format 
        if self.return_sparse_e:
            non_dig_e = np.where(~np.eye(edge_emb.shape[0],dtype=bool))
            edge_emb = edge_emb[non_dig_e]
            
        
        
        #convert label to one-hot encoding
        #res = np.zeros(n_label)
        #res[label] = 1
        #label = res
        return Graph(a=adjacency , x=node_emb , e=edge_emb , y= [ans_start, ans_end])

if __name__ == "__main__":
    path = "graph_ele_val.hdf5"
    dataset = DocDataset(path)
    #dataset = TUDataset('PROTEINS')
    print(dataset)
    print(type(dataset[0]))