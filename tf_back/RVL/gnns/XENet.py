from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras import backend as K
from spektral.layers import XENetConv, GlobalSumPool 
from spektral.layers.convolutional.message_passing import MessagePassing

import tensorflow as tf

class XENetGCN(Model):

    def __init__(self,n_labels= None, l2_reg=2.5e-4, activation="relu", dropout=0.25, dim=768, edge_emb_d = [1000, 1200, 1400, 1200]):
        super().__init__()
        reg = tf.keras.regularizers.l2(l2_reg)
        self.pre_edge = Dense(768, 'tanh')
        #self.x_emb = Embedding(edge_emb_d[2], int(dim/4))
        #self.y_emb = Embedding(edge_emb_d[1], int(dim/4))
        #self.x2_emb = Embedding(edge_emb_d[2], int(dim/4))
        #self.y2_emb = Embedding(edge_emb_d[3], int(dim/4))
        
        self.graph_conv_1 = XENetConv(dim, dim, dim, node_activation=activation, edge_activation="tanh", kernel_regularizer=reg, bia_regularizer=reg)
        self.graph_conv_2 = XENetConv(dim, dim, dim, node_activation=activation, edge_activation="tanh", kernel_regularizer=reg, bia_regularizer=reg)
        self.pool = GlobalSumPool()
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)

        self.dense = Dense(n_labels, 'softmax')
        
        #readout function will merge the node embedding and edge embedding to form final feature
    def readout(self, nodes, edges,index=0):
        n =  tf.shape(nodes)[0] - 1
        
        x_i = nodes[index]
        x_j = K.concatenate([nodes[:index], nodes[index+1:]], axis=0)
         
        
        x_i = tf.ones([n,1])*x_i
        e_ij = edges[index*n: (index+1)*n]
        ji_indices = tf.range(index, n**2, n )
        
        e_ji = tf.gather(edges, ji_indices)

        # Concatenate the features and feed to first MLP
        stack_ij = K.concatenate(
            [x_i, x_j, e_ij, e_ji], axis=-1)
        
        return stack_ij
        
        


    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        elif len(inputs) == 3:
            x, a, _ = inpits
        else:
            x, a, e, _ = inputs
        
        
        
        #e = K.concatenate([self.x_emb(e[:,0]), self.y_emb(e[:,1]), self.x_emb(e[:,2]), self.y_emb(e[:,2])], axis=-1)
        e = self.pre_edge(e)
        x = self.dropout_1(x)
        e = self.dropout_1(e)
        x,e = self.graph_conv_1([x,a,e])
        
        
        x = self.dropout_2(x)
        e = self.dropout_2(e)
        x,e = self.graph_conv_2([x,a,e])
        
     
        x = self.readout(x, e)
        x = self.pool(x)
        
        x = self.dense(x)
     
    
        return x
