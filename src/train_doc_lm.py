import os
import argparse
from pydoc import describe

from pandas import describe_option

# import torch.optim
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import pickle
from utils.params import Params
import LMs
# from torch_geometric.transforms import NormalizeFeatures
import dataload
from LMs import trainer
from utils import util_trainer 
import LMs

def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/lm.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, load data; prepare output_dim/num_labels, id2label, label2id for section3; 
    mydata = dataload.setup(params)

    # section 3, objective function and output dim
    params.criterion = util_trainer.get_criterion(params)

    # section 4, model, loss function, and optimizer
    if bool(params.continue_train):
        model_params = pickle.load(open(os.path.join(params.continue_with_model,'config.pkl'),'rb'))
        model = LMs.setup(model_params).to(params.device)
        model.load_state_dict(torch.load(os.path.join(params.dir_name,'model')))
    else:
        model = LMs.setup(params).to(params.device)

    # section 5, train and evaluate
    trainer.train(params, model, mydata)


