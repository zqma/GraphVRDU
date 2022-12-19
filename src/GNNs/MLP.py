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
        if self.opt.task_type == 'node-classify':
            self.mlp1 = nn.Linear(opt.node_dim, opt.hidden_dim_1)
        else:
            self.mlp1 = nn.Linear(opt.node_dim *2, opt.hidden_dim_1)
        self.mlp2 = nn.Linear(opt.hidden_dim_1, opt.hidden_dim_2)

        self.multilabel = nn.Linear(opt.hidden_dim_2, opt.output_dim)
        self.single = nn.Linear(opt.hidden_dim_2, 1)
        

    def forward(self,data):
        if self.opt.task_type in ['link-binary','direct-classify','neib-regression','joint']:
            return self.edge_prediction(data)
        elif self.opt.task_type == 'node-classify':
            return self.node_classifier(data)


    def edge_prediction(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_src = x[edge_index[0]]
        x_tgt = x[edge_index[1]]

        x = torch.cat(( x_src, x_tgt),-1)

        x = self.mlp1(x)
        x = x.relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.mlp2(x)
        x = x.relu()
        x = F.dropout(x,p=self.opt.dropout, training = self.training)

        if self.opt.task_type == 'link-binary': # binary classification
            y = self.single(x)
            return F.sigmoid(y)
        elif self.opt.task_type == 'direct-classify':   # multi-class classification
            y = self.multilabel(x)
            return F.log_softmax(x, dim=-1)
        elif self.opt.task_type == 'neib-regression':   # numeric regression
            y = self.single(x)
            return y
        elif self.opt.task_type == 'joint':
            y1 = y = self.multilabel(x)
            y1 = F.log_softmax(x, dim=-1)
            y2 = self.single(x)
            return y1,y2


    def node_classifier(self, data):
        x = data.x

        # x = self.encoder(x)
        x = self.mlp1(x)
        x = x.relu()
        x = F.dropout(x, p=self.opt.dropout, training = self.training)
        x = self.mlp2(x)
        x = x.relu()
        x = F.dropout(x,p=self.opt.dropout, training = self.training)

        x = self.multilabel(x)

        # return x
        return F.log_softmax(x, dim=-1)




