import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report


def my_loss(output, y):
    # loss = torch.mean((output - y)**2)
    loss = -(output.log()* y * 99 + (1-y)*(1-output).log()*1).mean()
    return loss

# get criterion and determine the output dim
def get_criterion(opt):
    if opt.task_type in ['node-classify', 'direct-classify','token-classifier']:
        if opt.task_type == 'node-classify':
            opt.output_dim = 4
        elif opt.task_type== 'direct-classify': 
            opt.output_dim = 8
        elif opt.task_type=='token-classifier':
            opt.output_dim = opt.num_labels
        return torch.nn.CrossEntropyLoss()
        
    elif opt.task_type == 'link-binary':
        opt.output_dim = 1
        # class_weights=torch.tensor([0.9,0.1],dtype=torch.float)
        return torch.nn.BCEWithLogitsLoss()
        # return my_loss
    elif opt.task_type == 'neib-regression':
        opt.output_dim = 1
        return torch.nn.L1Loss()
        # return torch.nn.MSELoss()
    else:
        raise Exception('task type error, not supported:{}'.format(opt.task_type)) 

def get_target(opt,data):
    if opt.task_type == 'link-binary':
        target = data.y.to(opt.device)
    elif opt.task_type == 'node-classify':
        target = data.y_nrole.to(opt.device)
    elif opt.task_type == 'neib-regression':
        target = data.y_dist
    elif opt.task_type == 'direct-classify':
        target = data.y_direct
    return target

def train(opt, model, mydata):
    # 1 data loader
    loader_train = DataLoader(mydata.train_graphs, batch_size=6)
    loader_test = DataLoader(mydata.test_graphs, batch_size=2)
    print(loader_train, loader_test)

    # 2 optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, weight_decay=5e-4)

    for epoch in range(opt.epochs):
        # train mode
        model.train()
        for _i, graph in enumerate(loader_train,start=0):
            optimizer.zero_grad()  # Clear gradients.
            out = model(graph)  # Perform a single forward pass.
            tgt = get_target(opt,graph)
            if opt.output_dim == 1: out = out.view(-1)
            loss = opt.criterion(out,tgt)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        # test
        model.eval()
        preds,tgts = predict_all_graphs(opt, model,loader_test)
        
        # taste the pred
        # print(preds[:8])
        # print(tgts[:8])
        if opt.task_type == 'neib-regression':
            mse = test_mse(preds,tgts)
            print('MSE:',mse)
        else:
            # val_acc = test_accu(preds, tgts)
            res_dict = evaluate(preds,tgts,True)
            print(res_dict)
    return loss


# return the predicted labels
def predict_one_graph(model, graph):
    pred = model(graph)
    pred = pred.argmax(dim=-1)
    return pred

def predict_all_graphs(opt, model, dataloader_test):
    outputs, targets = [],[]
    for _ii, data in enumerate(dataloader_test, start=0):
        output = predict_one_graph(model, data)
        target = get_target(opt, data).to(opt.device)

        outputs.append(output)
        targets.append(target)
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    return outputs,targets


def test_accu(y_pred,y_truth):
    test_correct = y_pred == y_truth # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(len(y_truth))  # Derive ratio of correct predictions.
    print(str(test_correct.sum()) +' / '+ str(len(y_truth)))
    return test_acc

def test_mse(y_pred,y_truth):
    return torch.nn.MSELoss()(y_pred, y_truth)


def evaluate(outputs, targets, print_confusion=False):

    # n_total,num_classes = outputs.shape
    # 2) move to cpu to convert to numpy
    output = outputs.cpu().numpy()
    target = targets.cpu().numpy()

    # confusion = confusion_matrix(output, target)
    f1 = f1_score(target, output, average='weighted')
    precision,recall,fscore,support = precision_recall_fscore_support(target, output, average='weighted')
    acc = accuracy_score(target, output)
    performance_dict = {'num':len(output),'acc': round(acc,3), 'f1': round(f1,3), 'precision':round(precision,3),'recall':round(recall,3)}
    if print_confusion: print(classification_report(target, output))

    return performance_dict