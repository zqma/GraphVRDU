from spektral.data import BatchLoader, DisjointLoader
from graph_dataset import *
from gnns import GAT, crystal, XENet
from spektral.transforms import GCNFilter
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from spektral.layers import GCNConv, GeneralConv, GATConv, CrystalConv,GlobalSumPool 
from spektral.models import GeneralGNN
from spektral.transforms import LayerPreprocess
from sklearn.model_selection import KFold, cross_val_score
from tensorflow import keras 

import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class MyFirstGNN(Model):

    def __init__(self, channels = None,n_labels= None, l2_reg=2.5e-4, activation="prelu", dropout=0.25):
        super().__init__()
        reg = tf.keras.regularizers.l2(l2_reg)
        self.graph_conv_1 = GeneralConv(n_hidden[0], activation=activation, kernel_regularizer=reg)
        self.graph_conv_2 = GeneralConv(n_hidden[1], activation=activation, kernel_regularizer=reg)
        self.pool = GlobalSumPool()
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs
    
        x = self.graph_conv_1([x,a])
        x = self.dropout_1(x)
        
        x = self.graph_conv_2([x,a])
        x = self.dropout_2(x)
        
        x = self.pool(x)
        x = self.dense(x)

        return x
        
class baseline(Model):
    def __init__(self,):
        super().__init__()
        self.dense_1 = Dense(256, "relu")
        self.dense_2 = Dense(16, "relu")
        self.pool = GlobalSumPool()
        
    def call(self, inputs):
        x, _, _, _ = inputs
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.pool(x)
        
        return x
        
        
    

if __name__ == "__main__":
    #load and filter dataset
    configs = {
    "GATConv" : {"layer": GATConv, "return_sparse_a": False, "transforms": LayerPreprocess(GATConv)},
    "crystal": {"layer": CrystalConv, "return_sparse_a": True, "transforms": LayerPreprocess(CrystalConv), "return_edge": True}
    }
    model = "crystal"
    config = configs[model]
    path = "graph_ele.hdf5"
    dataset = DocDataset(path, return_sparse_a = config["return_sparse_a"], transforms = config["transforms"], return_edge= config["return_edge"], use_feaure="text") #[1000, 1200,1400, 1200]

    #split dataset into train and test
    k_fold = KFold(n_splits=5, random_state=2022, shuffle=True)
    es = EarlyStopping(
    monitor='loss', 
    patience=5, 
    min_delta=0.05, 
    mode='min', 
    restore_best_weights=True
    )  
    test_acc = []
    i = 1
    for train_indices, test_indices in k_fold.split(dataset):
        print("-"*10 + str(i) + "-"*10)
        i += 1
        train_set, test_set = dataset[train_indices], dataset[test_indices]
        train_loader = DisjointLoader(train_set, batch_size=1)
        test_loader = DisjointLoader(test_set, batch_size=1)
        #initialize gcn model
        n_hidden = [128, 256]
        #model = MyFirstGNN(channels = n_hidden, n_labels = dataset.n_labels)
        #model = GAT.GraphAttNN(n_labels = dataset.n_labels, dropout=0.25, channels=n_hidden)
        model = XENet.XENetGCN(n_labels = dataset.n_labels)
        #model = baseline()
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer, 'categorical_crossentropy', ["categorical_accuracy"])
        model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=50, callbacks=[es], verbose=2
        )
        
        loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch, verbose=2)
        print('Test loss: {}'.format(loss))
        test_acc.append(np.round(loss[-1], 2))
        
    print("5-cross validation results: ", test_acc)

        
