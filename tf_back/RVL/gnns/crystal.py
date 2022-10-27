from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from spektral.layers import GCNConv, GeneralConv, GlobalSumPool 
from spektral.layers.convolutional.message_passing import MessagePassing

import tensorflow as tf

class CrystalGCN(Model):

    def __init__(self,n_labels= None, l2_reg=2.5e-4, activation="relu", dropout=0.25):
        super().__init__()
        reg = tf.keras.regularizers.l2(l2_reg)
        self.pre_edge = Dense(768, "tanh")
        self.graph_conv_1 = CrystalConv(activation=activation, kernel_regularizer=reg, bia_regularizer=reg)
        self.graph_conv_2 = CrystalConv(activation=activation, kernel_regularizer=reg, bia_regularizer=reg)
        self.pool = GlobalSumPool()
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)

        self.dense = Dense(n_labels, 'softmax')
        


    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        elif len(inputs) == 3:
            x, a, _ = inpits
        else:
            x, a, e, _ = inputs
        
        e = self.pre_edge(e)
        x = self.dropout_1(x)
        
        x = self.graph_conv_1([x,a,e])
        e = e + e*self.graph_conv_1.e
        
        
        x = self.dropout_2(x)
        x = self.graph_conv_2([x,a,e])
        e = e + e*self.graph_conv_2.e
        
        #e = self.pool(x)
        x = self.pool(x)
        
        #x = K.concatenate([e,x], axis=-1)
        x = self.dense(x)
        #x = self.dense_2(x)
    
        return x
        
#customize crystalconv layer
class CrystalConv(MessagePassing):
    r"""
    A crystal graph convolutional layer from the paper
    > [Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties](https://arxiv.org/abs/1710.10324)<br>
    > Tian Xie and Jeffrey C. Grossman
    **Mode**: single, disjoint, mixed.
    **This layer expects a sparse adjacency matrix.**
    This layer computes:
    $$
        \x_i' = \x_i + \sum\limits_{j \in \mathcal{N}(i)} \sigma \left( \z_{ij}
        \W^{(f)} + \b^{(f)} \right) \odot \g \left( \z_{ij} \W^{(s)} + \b^{(s)}
        \right)
    $$
    where \(\z_{ij} = \x_i \| \x_j \| \e_{ji} \), \(\sigma\) is a sigmoid
    activation, and \(g\) is the activation function (defined by the `activation`
    argument).
    **Input**
    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`.
    - Edge features of shape `(num_edges, n_edge_features)`.
    **Output**
    - Node features with the same shape of the input.
    **Arguments**
    - `activation`: activation function;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(
        self,
        aggregate="sum",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype,
        )
        channels = input_shape[0][-1]
        self.dense_f = Dense(channels, activation="sigmoid", **layer_kwargs)
        self.dense_s = Dense(channels, activation=self.activation, **layer_kwargs)
        self.dense_e = Dense(channels, activation="sigmoid", **layer_kwargs)
        self.built = True

    def message(self, x, e=None):
        x_i = self.get_targets(x)
        x_j = self.get_sources(x)

        to_concat = [x_i, x_j]
        if e is not None:
            to_concat += [e]
        z = K.concatenate(to_concat, axis=-1)
        output = self.dense_s(z) * self.dense_f(z)
        self.e = self.dense_e(z)
        return output
  

    def update(self, embeddings, x=None):
        return x + embeddings
