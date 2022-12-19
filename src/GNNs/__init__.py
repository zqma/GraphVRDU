
from GNNs.GAT import GAT
from GNNs.GCN import GCN
from GNNs.GAE import GAE
from GNNs.MLP import MLP
from GNNs.GraphSage import GraphSage

def setup(opt):
    print('network:' + opt.network_type)
    if opt.network_type == 'gat':
        model = GAT(opt)
    elif opt.network_type == 'gcn':
        model = GCN(opt)
    elif opt.network_type == 'gae':
        model = GAE(opt)
    elif opt.network_type == 'mlp':
        model = MLP(opt)
    elif opt.network_type == 'graphsage':
        model = GraphSage(opt)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))
    return model
