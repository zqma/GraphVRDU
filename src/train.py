import os

import torch.optim
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graph_models.GNN import MyGCN

dataset = Planetoid(root='../data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

graph = dataset[0]  # Get the first graph object.

print()
print(graph)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {graph.num_nodes}')
print(f'Number of edges: {graph.num_edges}')
print(f'Average node degree: {graph.num_edges / graph.num_nodes:.2f}')
print(f'Number of training nodes: {graph.train_mask.sum()}')
print(f'Training node label rate: {int(graph.train_mask.sum()) / graph.num_nodes:.2f}')
print(f'Has isolated nodes: {graph.has_isolated_nodes()}')
print(f'Has self-loops: {graph.has_self_loops()}')
print(f'Is undirected: {graph.is_undirected()}')

# start the model

model = MyGCN(input_dim=graph.num_features,   # 1433
            hidden_dim=16,
            output_dim=dataset.num_classes)   # 7

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)

def train(epoch_num=100):
    model.train()
    for epoch in range(epoch_num):
        optimizer.zero_grad()  # Clear gradients.
        out = model(graph)  # Perform a single forward pass.
        loss = criterion(out[graph.train_mask],
                     graph.y[graph.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    return loss


def test_accu(model,graph):
    model.eval()
    out = model(graph)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(graph.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

train(20)
accu = test()
print(accu)