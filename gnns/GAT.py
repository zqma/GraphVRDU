from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GeneralConv, GATConv, GlobalSumPool 

import tensorflow as tf

class GraphAttNN(Model):

    def __init__(self, channels = None,n_labels= None, l2_reg=2.5e-4, activation="elu", n_attention_heads=8, dropout=0.25):
        super().__init__()
        reg = tf.keras.regularizers.l2(l2_reg)
        self.graph_conv_1 = GATConv(channels[0], attention_heads = n_attention_heads, activation=activation, kernel_regularizer=reg, attn_kernel_regularizer=reg, bia_regularizer=reg,                               concat_heads=True)
        self.graph_conv_2 = GATConv(channels[1], attention_heads = 1, activation=activation, kernel_regularizer=reg, attn_kernel_regularizer=reg, bias_regularizer=reg, 
                            concat_heads=False)
        self.pool = GlobalSumPool()
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs
        
        x = self.dropout_1(x)
        x = self.graph_conv_1([x,a])
        
        x = self.dropout_2(x)
        x = self.graph_conv_2([x,a])
        
        
        x = self.pool(x)
        x = self.dense(x)
    
        return x