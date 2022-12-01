# -*- coding: utf-8 -*-

import torch.nn as nn
from models.base_model import BaseModel

from transformers import RobertaModel, RobertaConfig

class RobertaClassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(RobertaClassifier, self).__init__()
        self.opt = opt
        self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        self.roberta = RobertaModel.from_pretrained(opt.roberta_dir, config=self.config)


        self.classifier = nn.Sequential(
            nn.Linear(self.opt.input_dim,self.opt.input_dim),   # hidden dim
            nn.ReLU(),
            nn.Dropout(self.opt.dropout),
            nn.Linear(self.opt.input_dim, self.opt.reader.nb_classes),
            nn.Sigmoid()
        )

        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False


    def forward(self,input_ids, attention_mask):
        outputs = self.roberta(input_ids = input_ids, attention_mask = attention_mask)
        # use the first token CLS state for sentence-level prediction
        # last_hidden_state_cls = outputs[0][:,0,:] # (batch_size, seq_len, dim)  => (batch_size, 1, dim)
        hidden_state = outputs[0]  # (batch_size, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (batch_size, dim) for first token CLS state
        # feet input to classifier to compute logits
        logits = self.classifier(pooled_output)

        return logits

