import os
import argparse
from pydoc import describe

from pandas import describe_option

# import torch.optim
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from utils.params import Params
from torch_geometric.loader import DataLoader
# from torch_geometric.transforms import NormalizeFeatures
import dataload



import GNNs

# print(f'Number of training nodes: {graph.train_mask.sum()}')
# print(f'Training node label rate: {int(graph.train_mask.sum()) / graph.num_nodes:.2f}')
# print(f'Has isolated nodes: {graph.has_isolated_nodes()}')
# print(f'Has self-loops: {graph.has_self_loops()}')
# print(f'Is undirected: {graph.is_undirected()}')

def train(loader_train,epoch_num=100):
    model.train()
    for epoch in range(epoch_num):
        for _i, graph in enumerate(loader_train,start=0):
            optimizer.zero_grad()  # Clear gradients.
            out = model(graph)  # Perform a single forward pass.
            loss = criterion(out,
                        graph.y_nrole)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            # test
            val_acc = test_accu(graph, model)
            print(val_acc)
    return loss


def test_accu(graph,model):
    model.eval()
    out = model(graph)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred == graph.y_nrole  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(graph.x.shape[0])  # Derive ratio of correct predictions.
    return test_acc

def parse_args():
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = 'config/gnn.ini')
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args() # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, load data;
    mydata = dataload.setup(params)
    print(f'Number of train graphs: {len(mydata.train_graphs)}')
    train_loader = DataLoader(mydata.train_graphs, batch_size=6)

    # section 3, mannually add features
    params.input_dim = params.node_dim
    params.output_dim = mydata.node_num_class

    # section 4, model, loss function, and optimizer
    model = GNNs.setup(params).to(params.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)

    # section 5, train and evaluate
    train(train_loader,10)

    print(f'Number of test graphs: {len(mydata.test_graphs)}')
    test_loader = DataLoader(mydata.test_graphs, batch_size=6)
    accu = test_accu(test_loader.batch(),model)
    print(accu)



