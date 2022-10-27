# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

from transformers import DistilBertModel, DistilBertConfig
#DistilBertForSequenceClassification

class DistilBERTClassifier(BaseModel):
    def __init__(self, opt, freeze_bert=False):
        super(DistilBERTClassifier, self).__init__()
        self.opt = opt
        self.config = DistilBertConfig(n_heads=opt.n_heads, n_layers=opt.n_layers, attention_dropout=0.1)
        # self.distilbert = DistilBertModel(self.config)
        self.distilbert = DistilBertModel.from_pretrained(opt.distilbert_dir)
        # self.distilbertseq = DistilBertForSequenceClassification.from_pretrained(opt.distilbert_dir)
        self.classifier = nn.Sequential(
            nn.Linear(self.opt.input_dim,self.opt.hidden_dim),# hidden_dim
            nn.ReLU(),
            nn.Dropout(self.opt.dropout),
            nn.Linear(self.opt.hidden_dim, self.opt.reader.nb_classes),
            nn.Sigmoid()
        )

        # freeze the bert model
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False



    def forward(self,input_ids, attention_mask):
        outputs = self.distilbert(input_ids = input_ids, attention_mask = attention_mask)
        # this is to use CLS (the first token state) to predict the sentence-level classification
        hidden_state = outputs[0] # (batch_size, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (batch_size, dim) for first token CLS state
        # feet input to classifier to compute logits
        logits = self.classifier(pooled_output)

        return logits

