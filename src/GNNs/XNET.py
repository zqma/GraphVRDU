import torch
import torch.nn as nn
from torch_geometric.nn import 
import torch.nn.functional as F

class GAE(torch.nn.Module):
    def __init__(self, opt):
        super(GAE, self).__init__()
        self.opt = opt
        self.conv1 = GCNConv(opt.input_dim, opt.hidden_dim_1)
        self.conv2 = GCNConv(opt.hidden_dim_1, opt.hidden_dim_2)


    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0).nonzero(as_tuple=False).t()