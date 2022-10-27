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

        self.gat1 = GATConv(opt.input_dim, opt.hidden_dim, heads=self.in_head, dropout=opt.dropout)
        self.gat2 = GATConv(opt.hidden_dim * self.in_head, opt.output_dim, heads=self.out_head)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.gat2(x, edge_index)
        return x
        # return F.log_softmax(x, dim=-1)

