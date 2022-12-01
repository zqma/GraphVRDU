# -*- coding: utf-8 -*-

import torch.nn as nn

# from transformers import RobertaModel, RobertaConfig
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification

class LayoutLMTokenclassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(LayoutLMTokenclassifier, self).__init__()
        self.opt = opt
        # self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        self.layoutlm = AutoModelForTokenClassification.from_pretrained(opt.layoutlm_dir, num_labels=opt.num_labels, label2id=opt.label2id, id2label=opt.id2label)

        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    def forward(self,input_ids, bbox, attention_mask, pixel_values, labels):
        outputs = self.layoutlm(input_ids = input_ids, bbox = bbox, attention_mask = attention_mask, pixel_values = pixel_values, labels = labels)
        return outputs

