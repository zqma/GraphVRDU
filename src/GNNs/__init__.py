
from GNNs.GAT import GAT
from GNNs.GCN import GCN


def setup(opt):
    print('network:' + opt.network_type)
    if opt.network_type == 'gat':
        model = GAT(opt)
    elif opt.network_type == 'gcn':
        model = GCN(opt)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))
    return model
