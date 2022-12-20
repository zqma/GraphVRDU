import os
import argparse
from pydoc import describe

from pandas import describe_option

# import torch.optim
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from utils.params import Params

# from torch_geometric.transforms import NormalizeFeatures
import dataSetup
import GNNs
from GNNs import trainer


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
    mydata = dataSetup.setup(params)
    # if params.task_type == 'link-binary':
    #     mydata.balance_edges()
    # print(mydata.)

    # section 3, objective function and output dim
    params.criterion = trainer.get_criterion(params)

    # section 4, model, loss function, and optimizer
    model = GNNs.setup(params).to(params.device)

    # section 5, train and evaluate
    trainer.train(params, model, mydata)


