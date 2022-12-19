import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class GraphSage(torch.nn.Module):
    def __init__(self, opt):
        super(GraphSage, self).__init__()
        # self.layer_num = opt.layer_num
        # self.encoder = nn.Linear(opt.input_dim, opt.hidden_dim)
        self.opt = opt

        self.conv1 = SAGEConv(opt.node_dim, opt.hidden_dim_1)
        self.conv2 = SAGEConv(opt.hidden_dim_1, opt.hidden_dim_2)

        self.multi_dim4node = SAGEConv(opt.hidden_dim_1, opt.output_dim)

        self.multi_dim4edge = nn.Linear(opt.hidden_dim_2 * 2 , opt.output_dim)
        self.single_dim4edge = nn.Linear(opt.hidden_dim_2 * 2, 1)

    def forward(self, data):
        if self.opt.task_type in ['link-binary','direct-classify','neib-regression','joint']:
            return self.edge_prediction(data)
        elif self.opt.task_type == 'node-classify':
            return self.node_classifier(data)

    def edge_prediction(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = self.encode(data)
        # then, get two node reps
        x_src = x[edge_index[0]]
        x_tgt = x[edge_index[1]]
        # edge rep
        x = torch.cat(( x_src, x_tgt),-1)
        
        if self.opt.task_type == 'link-binary': # binary classification
            y = self.single_dim4edge(x)
            return F.sigmoid(y)
        elif self.opt.task_type == 'direct-classify':   # multi-class classification
            y = self.multi_dim4edge(x)
            return F.softmax(x, dim=-1)
        elif self.opt.task_type == 'neib-regression':   # numeric regression
            y = self.single_dim4edge(x)
            return y
        elif self.opt.task_type == 'joint':
            y1 = y = self.multi_dim4edge(x)
            y1 = F.softmax(x, dim=-1)
            y2 = self.single_dim4edge(x)
            return y1,y2


    def node_classifier(self, data):
        x, edge_index = data.x, data.edge_index
        rep = self.conv1(x, edge_index).relu()
        rep = F.dropout(rep, p=self.opt.dropout, training = self.training)
        rep = self.multi_dim4node(rep, edge_index)
        # return x
        return F.softmax(rep, dim=-1)



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
