import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, opt):
        super(GCN, self).__init__()
        # self.layer_num = opt.layer_num
        # self.encoder = nn.Linear(opt.input_dim, opt.hidden_dim)
        self.opt = opt
        self.conv1 = GCNConv(opt.input_dim, opt.hidden_dim_l)
        self.conv2 = GCNConv(opt.hidden_dim_l, opt.hidden_dim_m)

        self.classifier = GCNConv(opt.hidden_dim_l,opt.output_dim)
        self.binary_or_numeric = GCNConv(opt.hidden_dim, 1)

        self.edge_classifier = nn.Linear(opt.hidden_dim_m * 2, opt.output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = self.encoder(x)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.classifier(x, edge_index)
        prob = F.log_softmax(x, dim=-1) # label probability distribution
        return prob

    def edge_direct_classification(self, data):
        x, edge_index = data.x, data.edge_index
        # size, and dist, e.g.. [13,45], 0.6
        # x: [num_nodes, num_features]
        # edge_index: [2* num_edge]

        x = self.encode(data)   # => [num_node, hidden_dim_m]

        e = torch.cat((x[edge_index[0]], x[edge_index[1]]),-1)
        y = self.edge_classifier(e)
        
        # return x
        return F.log_softmax(x, dim=-1)



    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        y = (z[edge_label_index[0]] * z[edge_label_index[1]])   # element wise product
        y = y.sum(dim=-1)   # some of the dim parameters;  
        return y

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0).nonzero(as_tuple=False).t()
