from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report
import time
import numpy as np
import os
import pickle

def temp_a_dir(opt):
    # Temp file for storing the best model
    temp_file_name = str(int(np.random.rand() * int(time.time())))
    opt.best_model_file = os.path.join('tmp', temp_file_name)


def train(opt, model, mydata):
    # 1 data loader
    loader_train = DataLoader(mydata.train_dataset, batch_size=opt.batch_size,shuffle=True)
    loader_test = DataLoader(mydata.test_dataset, batch_size=opt.batch_size)

    print(loader_train, loader_test)

    # 2 optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr, weight_decay=5e-4)
    # 3 training
    best_f1 = 0.0
    create_save_dir(opt)    # prepare dir for saving best models
    for epoch in range(opt.epochs):    
        # train mode
        model.train()
        for batch in tqdm(loader_train, desc = 'Training'):
            optimizer.zero_grad()  # Clear gradients.
            outputs = predict_one_batch(opt,model,batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()  # Update parameters based on gradients.
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
        # test
        model.eval()
        outputs,tgts = predict_all_batches(opt, model,loader_test)    
        # res_dict = evaluate(outputs.logits,tgts)
        res_dict = compute_metrics([outputs.logits,tgts])
        print(res_dict)

        if res_dict['f1']>best_f1:
            save_model(opt, model,res_dict)
            best_f1 = res_dict['f1']
            print('The best model saved with f1:', best_f1)
        


# for backpropagation use, so define the input variables
def predict_one_batch(opt, model, batch):
    if opt.task_type == 'token-classifier':
        input_ids = batch['input_ids'].to(opt.device)
        attention_mask = batch['attention_mask'].to(opt.device)
        bbox = batch['bbox'].to(opt.device)
        labels = batch['labels'].to(opt.device)
        pixel_values = batch['pixel_values'].to(opt.device)
        
        # outputs = model(**batch)
        outputs = model(input_ids = input_ids, 
            bbox = bbox, 
            attention_mask = attention_mask, 
            pixel_values = pixel_values, 
            labels = labels
        )
    
    return outputs

# for evaluation use (all batche inference)
def predict_all_batches(opt,model,dataloader):
    outputs, targets = [],[]
    for _ii, data in enumerate(dataloader, start=0):
        output = predict_one_batch(model, data)
        target = data['labels'].to(opt.device)

        outputs.append(output)
        targets.append(target)
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    return outputs,targets


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

def compute_metrics(p,return_entity_level_metrics=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)   # -1 or 2

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def create_save_dir(params):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

     # Create model dir
    params.dir_name = '_'.join([params.network_type,params.dataset_name,str(round(time.time()))[-6:]])
    dir_path = os.path.join('tmp', params.dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)  

    params.export_to_config(os.path.join(dir_path, 'config.ini'))
    pickle.dump(params, open(os.path.join(dir_path, 'config.pkl'), 'wb'))


# Save the model
def save_model(params, model, performance_str):
    # 1) save the learned model (model and the params used)
    torch.save(model.state_dict(), os.path.join(params.dir_path, 'model'))

    # 2) Write performance string
    eval_path = os.path.join(params.dir_path, 'eval')
    with open(eval_path, 'w') as f:
        f.write(performance_str)

