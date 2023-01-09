import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report
import os, pickle
import time


def joint_loss(opt,outputs,labels,t_lambda=0.5):
    # loss = torch.mean((output - y)**2)
    crit1,crit2 = opt.criterion
    loss = opt.t_lambda*crit1(outputs[0],labels[0]) + (1.0-opt.t_lambda)*crit2(outputs[1],labels[1])
    return loss


# get criterion and determine the output dim
def get_criterion(opt):
    if opt.task_type in ['node-classify', 'direct-classify','token-classifier']:
        if opt.task_type == 'node-classify':
            opt.output_dim = opt.num_labels
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
    elif opt.task_type == 'dist-regression':
        opt.output_dim = 1
        return torch.nn.L1Loss()
        # return torch.nn.MSELoss()
    elif opt.task_type == 'joint':
        opt.output_dim = 1
        return [torch.nn.CrossEntropyLoss(),torch.nn.L1Loss()]
    else:
        raise Exception('task type error, not supported:{}'.format(opt.task_type)) 

def get_target(opt,batch):
    if opt.task_type == 'link-binary':
        target = batch.y.to(opt.device)
    elif opt.task_type == 'node-classify':
        target = batch.y_nrole.to(opt.device)
    elif opt.task_type == 'dist-regression':
        target = batch.y_dist.to(opt.device)
    elif opt.task_type == 'direct-classify':
        target = batch.y_direct.to(opt.device)
    elif opt.task_type == 'joint':
        target = [batch.y_direct.to(opt.device),batch.y_dist.to(opt.device)]
    return target

def get_loss(opt,outputs,labels):
    if opt.task_type == 'joint':
        loss = joint_loss(opt,outputs,labels)
    else:
        if opt.output_dim == 1: outputs = outputs.view(-1)
        loss = opt.criterion(outputs,labels)  # Compute the loss solely based on the training nodes.
    return loss

def train(opt, model, mydata):
    # 1 data loader
    loader_train = DataLoader(mydata.train_graphs, batch_size=6)
    loader_test = DataLoader(mydata.test_graphs, batch_size=2)
    print(loader_train, loader_test)

    # 2 optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, weight_decay=5e-4)

    opt.dir_path = create_save_dir(opt)    # prepare dir for saving best models
    best_loss = 99999.9
    for epoch in range(opt.epochs):
        # train mode
        model.train()
        for _i, graph in enumerate(loader_train,start=0):
            optimizer.zero_grad()  # Clear gradients.

            outputs = predict_one_batch(opt,model,graph)
            labels = get_target(opt,graph)
            loss = get_loss(opt,outputs,labels)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        
        # test
        model.eval()

        preds,tgts,val_loss = predict_all_batches(opt, model,loader_test)
        print(f'Epoch: {epoch:03d}, Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            # save_model(opt, model,res_dict)
            print('The best model saved with loss:', val_loss)  
            # update the pre-trained vectors
            output_hidden_vect(opt,model,loader_test,'test')
            output_hidden_vect(opt,model,loader_train,'train')
        # taste the pred
        # print(preds[:8])
        # print(tgts[:8])
        if opt.task_type in ['dist-regression','joint']:
            print('MSE is the val loss',)
        else:
            # val_acc = test_accu(preds, tgts)
            res_dict = evaluate(preds,tgts,True)
            print(res_dict)
    return val_loss


# only return pred logits, for back propagation
def predict_one_batch(opt,model, graph):
    graph = graph.to(opt.device)
    pred = model(graph)
    return pred

# return pred labels, labels, and loss
def predict_all_batches(opt, model, dataloader_test):
    outputs, targets, val_loss = [],[],0

    for _ii, batch in enumerate(dataloader_test, start=0):
        preds = predict_one_batch(opt,model, batch)
        target = get_target(opt, batch)
        val_loss += get_loss(opt,preds,target)
        if opt.task_type in ['dist-regression','joint']:        
            continue

        preds = torch.argmax(preds, dim=-1)
        outputs.append(preds)
        targets.append(target)

    # numeric return
    if opt.task_type in ['dist-regression','joint']: 
        return outputs,targets,val_loss
    # category return
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    return outputs,targets,val_loss


def output_hidden_vect(opt,model,dataloader,split='test'):
    doc_ids, seg_ids,seg_vects = [], [],[]
    
    for _ii, batch in enumerate(dataloader, start=0):
        seg_ids.append(batch.seg_id)
        doc_ids.append(batch.doc_id)
        batch = batch.to(opt.device)
        vects = model.encode(batch)
        seg_vects.append(vects)
    doc_ids = torch.cat(doc_ids)
    seg_ids = torch.cat(seg_ids)
    seg_vects = torch.cat(seg_vects)
    
    pairs = zip(doc_ids.detach().cpu().numpy(),
        seg_ids.detach().cpu().numpy(),
        seg_vects.detach().cpu().numpy()
    )
    vect_path = os.path.join(opt.dir_path, opt.network_type+'seg_vect_'+split)
    with open(vect_path, 'w',encoding='utf8') as f:
        for docid,segid,vect in pairs:
            f.write(str(docid)+'\t'+str(segid)+'\t'+str(vect).replace('\n','')+'\n')


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

def create_save_dir(params):
    if not os.path.exists('tmp_dir'):
        os.mkdir('tmp_dir')

     # Create model dir
    dir_name = '_'.join([params.network_type,params.dataset_name,str(round(time.time()))[-6:]])
    dir_path = os.path.join('tmp_dir', dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)  

    params.export_to_config(os.path.join(dir_path, 'config.ini'))
    pickle.dump(params, open(os.path.join(dir_path, 'config.pkl'), 'wb'))
    return dir_path

# Save the model
def save_model(params, model, performance_str):
    # 1) save the learned model (model and the params used)
    torch.save(model.state_dict(), os.path.join(params.dir_path, 'model'))

    # 2) Write performance string
    eval_path = os.path.join(params.dir_path, 'eval')
    with open(eval_path, 'w') as f:
        f.write(str(performance_str))