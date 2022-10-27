# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformer_models.base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, opt):
        self.opt = opt
        super(MLP, self).__init__()
        self.input_dims = opt.embed_dim * opt.max_seq_len
        embedding_matrix = torch.tensor(opt.embedding.embedding_matrix, dtype=torch.float)
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.fc_out = nn.Sequential(
                            nn.Linear(self.input_dims, opt.hidden_dim_1),
                            nn.ReLU(),
                            nn.Dropout(opt.dropout),
                            nn.Linear(opt.hidden_dim_1,opt.hidden_dim_2),
                            nn.ReLU(),
                            nn.Dropout(opt.dropout),
                            nn.Linear(opt.hidden_dim_2, opt.reader.nb_classes)
                        )


    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.shape[0],-1)
        output = self.fc_out(x)
        return output
