from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,classification_report
import time
import numpy as np
import os
import pickle

from datasets import load_metric
metric = load_metric("seqeval")


def train(opt, model, mydata):
    # 1 data loader
    loader_train = DataLoader(mydata.train_dataset, batch_size=opt.batch_size,shuffle=True)
    loader_test = DataLoader(mydata.test_dataset, batch_size=opt.batch_size)

    print(loader_train, loader_test)

    # 2 optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr, betas=(0.9,0.999),eps=1e-08)
    # 3 training
    best_f1 = 0.0
    opt.dir_path = create_save_dir(opt)    # prepare dir for saving best models
    
    
    for epoch in range(opt.epochs):    
        print('epoch:',epoch,'/',str(opt.epochs))
        # train mode
        model.train()
        for batch in tqdm(loader_train, desc = 'Training'):
            optimizer.zero_grad()  # Clear gradients.
            outputs = predict_one_batch(opt,model,batch,eval=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()  # Update parameters based on gradients.
        
        # eval mode
        res_dict = test_eval(opt,model,loader_test)
        if res_dict['f1']>best_f1:
            save_model(opt, model,res_dict)
            best_f1 = res_dict['f1']
            print('The best model saved with f1:', best_f1)
    
    return best_f1


def test_eval(opt,model,loader_test):
    # test
    model.eval()
    preds,tgts, val_loss = predict_all_batches(opt, model,loader_test)
    print(f'val Loss: {val_loss:.4f}')

    # res_dict = evaluate(preds,tgts)
    if opt.task_type == 'docvqa':
        return
    
    res_dict = compute_metrics(opt, [preds,tgts])
    print(res_dict)

    return res_dict

# for backpropagation use, so define the input variables
def predict_one_batch(opt, model, batch, eval=False):
    if opt.task_type == 'token-classifier':
        input_ids = batch['input_ids'].to(opt.device)
        attention_mask = batch['attention_mask'].to(opt.device)
        bbox = batch['bbox'].to(opt.device)
        labels = batch['labels'].to(opt.device)
        pixel_values = batch['pixel_values'].to(opt.device)
        if eval:
            with torch.no_grad():
                outputs = model(
                    input_ids = input_ids, bbox = bbox, attention_mask = attention_mask, 
                    pixel_values = pixel_values, labels = labels)  
        else:
            outputs = model(
                input_ids = input_ids, bbox = bbox, attention_mask = attention_mask, 
                pixel_values = pixel_values, labels = labels)

    elif opt.task_type == 'docvqa':
        input_ids = batch['input_ids'].to(opt.device)
        attention_mask = batch['attention_mask'].to(opt.device)
        token_type_ids = batch['token_type_ids'].to(opt.device)
        bbox = batch['bbox'].to(opt.device)
        start_positions = batch['start_positions'].to(opt.device)
        end_positions = batch['end_positions'].to(opt.device)
        image = batch['image'].to(opt.device)

        if eval:
            with torch.no_grad():
                outputs = model(input_ids = input_ids, 
                    token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                    image = image, start_positions = start_positions, end_positions=end_positions)  
        else:
            outputs = model(input_ids = input_ids, 
                token_type_ids=token_type_ids, bbox = bbox, attention_mask = attention_mask, 
                image = image, start_positions = start_positions, end_positions=end_positions)
        

    return outputs

# for evaluation use (all batche inference)
def predict_all_batches(opt,model,dataloader):
        preds, tgts, val_loss = [],[], 0.0
        for _ii, batch in enumerate(dataloader, start=0):
            outputs = predict_one_batch(opt,model, batch, eval=True)
            predictions = torch.argmax(outputs.logits, dim=-1)
            if opt.task_type == 'docvqa':
                target = batch['start_positions']
            else:
                target = batch['labels']
            val_loss+=outputs.loss.item()

            preds.append(predictions)   # logits
            tgts.append(target)
        preds = torch.cat(preds)
        tgts = torch.cat(tgts)
        return preds,tgts,val_loss


# preds is the logits (label distribution);
def evaluate(preds, targets, print_confusion=False):
    # n_total,num_classes = outputs.shape
    # 2) move to cpu to convert to numpy
    preds = preds.cpu().numpy()
    target = targets.cpu().numpy()

    # confusion = confusion_matrix(output, target)
    f1 = f1_score(target, preds, average='weighted')
    precision,recall,fscore,support = precision_recall_fscore_support(target, preds, average='weighted')
    acc = accuracy_score(target, preds)
    performance_dict = {'num':len(preds),'acc': round(acc,3), 'f1': round(f1,3), 'precision':round(precision,3),'recall':round(recall,3)}
    if print_confusion: print(classification_report(target, preds))

    return performance_dict

def compute_metrics(opt, p,return_entity_level_metrics=False):
    predictions, labels = p
    # Remove ignored index (special tokens)
    true_predictions = [
        [opt.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [opt.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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

