import os
import argparse
from pydoc import describe

from pandas import describe_option

# import torch.optim
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from utils.params import Params
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling # negative sample for nodes or edges;
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score

import GNNs


def train(model,train_data, epoches):
    best_val_auc = final_test_auc = 0
    for epoch in range(1, epoches+1):
        loss = train_per_epoch(model,train_data)
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

def train_per_epoch(model,train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data)    # x, edge_index

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1))

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)


    out = model.decode(z, edge_label_index).view(-1)
    print(edge_label)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss



# def test_accu(model,graph):
#     model.eval()
#     out = model(graph)
#     pred = out.argmax(dim=1)  # Use the class with highest probability.
#     test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]  # Check against ground-truth labels.
#     test_acc = int(test_correct.sum()) / int(graph.test_mask.sum())  # Derive ratio of correct predictions.
#     return test_acc

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())



def parse_args():
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = 'config/gnn.ini')
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, load and parse parameters
    args = parse_args()
    # accomodate all params
    params = Params()
    params.parse_config(args.config_file)
    # params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, load the data
    # load the Cora dataset
    transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(params.device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.2, neg_sampling_ratio=1.0, is_undirected=True,
                      add_negative_train_samples=False),
    ])

    dataset = Planetoid('./data/Planetoid', name='Cora', transform=transform)
    # dataset = Planetoid(root='../data/Planetoid', name='Cora', transform=NormalizeFeatures())
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')    # 1433 (so many features)
    print(f'Number of classes: {dataset.num_classes}')

    train_data, val_data, test_data = dataset[0] # Get the first graph object.
    print(train_data)

    # section 3, mannually add features
    params.input_dim = train_data.num_features

    # section 4, model, loss function, and optimizer
    model = GNNs.setup(params).to(params.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)    # cannot use weight decay!!

    # section 5, train and evaluate
    epoch = 100
    train(model,train_data,epoch)
    accu = test(test_data)
    print(accu)

    z = model.encode(test_data)
    print(z)
    print(test_data.x.shape)    # 2708 * 1433


