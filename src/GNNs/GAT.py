import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GeneralConv
from torch_geometric.nn import CGConv   # crystal graph conv

import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, opt):
        super(GAT, self).__init__()
        self.opt = opt
        self.in_head = 8
        self.out_head = 1

        self.gat1 = GATConv(opt.input_dim, opt.hidden_dim_m, heads=self.in_head, dropout=opt.dropout)
        self.gat2 = GATConv(opt.hidden_dim_m * self.in_head, opt.hidden_dim_s, heads=self.out_head)
        
        self.classifier = GATConv(opt.hidden_dim_m * self.in_head, opt.output_dim, head = self.out_head)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.gat1(x, edge_index).relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.classifier(x, edge_index)
        return x
        # return F.log_softmax(x, dim=-1)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index).relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.gat2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        y = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        return y

    def decode_all(self, z):
        prob_adj = z @ z.t()
        #return the indices of a non-zero element
        return (prob_adj > 0).nonzero(as_tuple=False).t()
