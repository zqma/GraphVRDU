import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, opt):
        super(MLP, self).__init__()
        # self.layer_num = opt.layer_num
        # self.encoder = nn.Linear(opt.input_dim, opt.hidden_dim)
        self.opt = opt
        self.mlp1 = nn.Linear(opt.input_dim, opt.hidden_dim_l)
        self.mlp2 = nn.Linear(opt.hidden_dim_l, opt.hidden_dim_m)

        self.classifier = nn.Linear(opt.hidden_dim_m, opt.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(x.shape)
        print(edge_index.shape)

        # x = self.encoder(x)
        x = self.mlp1(x)
        x = x.relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.mlp2(x)
        x = x.relu()
        x = F.dropout(x,p=self.opt.dropout, training = self.training)

        x = self.classifier(x)

        # return x
        return F.log_softmax(x, dim=-1)


