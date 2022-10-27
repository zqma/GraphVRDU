from graph_models.GNN import MyGNN
# from graph_models.MLP import MLP
from graph_models.GAT import GAT
from graph_models.GCN import GCN


def setup(opt):
    print('network:' + opt.network_type)
    if opt.network_type == 'mygnn':
        model = MyGNN(opt)
    elif opt.network_type == 'gat':
        model = GAT(opt)
    elif opt.network_type == 'gcn':
        model = GCN(opt)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))
    return model
