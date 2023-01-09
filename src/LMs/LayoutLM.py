# -*- coding: utf-8 -*-

import torch.nn as nn

# from transformers import RobertaModel, RobertaConfig
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering


class LayoutLM4DocVQA(nn.Module):
    def __init__(self,opt,freeze_bert=False):
        super(LayoutLM4DocVQA, self).__init__()
        self.opt = opt

        self.layoutlm = AutoModelForQuestionAnswering.from_pretrained(opt.layoutlm_dir)

    # dif: token_type_ids; start and end positions -> labels; image -> no pixel_values
    def forward(self,input_ids, token_type_ids, bbox, attention_mask, image, start_positions, end_positions):
        outputs = self.layoutlm(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            bbox = bbox,
            attention_mask = attention_mask,
            image = image, 
            start_positions = start_positions,
            end_positions = end_positions
        )
        return outputs


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
        

    def forward(self,input_ids, bbox, attention_mask, pixel_values, labels, **args):
        outputs = self.layoutlm(input_ids = input_ids, bbox = bbox, attention_mask = attention_mask, pixel_values = pixel_values, labels = labels)
        # outputs = self.layoutlm(input_ids = input_ids, bbox = None, attention_mask = attention_mask, pixel_values = None, labels = labels)
        return outputs

