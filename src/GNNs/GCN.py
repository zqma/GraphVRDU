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
        self.conv1 = GCNConv(opt.input_dim, opt.hidden_dim)
        self.conv2 = GCNConv(opt.hidden_dim, opt.output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(x.shape)
        print(edge_index.shape)

        # x = self.encoder(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.conv2(x, edge_index)
        # return x
        return F.log_softmax(x, dim=-1)


