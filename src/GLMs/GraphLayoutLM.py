from transformers import LayoutLMv3Processor, LayoutLMv3Model
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch

class GraphLayoutLMTokenclassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(GraphLayoutLMTokenclassifier, self).__init__()
        self.opt = opt
        # self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        self.layoutlm = LayoutLMv3Model.from_pretrained(opt.layoutlm_dir, num_labels=opt.num_labels, label2id=opt.label2id, id2label=opt.id2label)
        self.dropout = nn.Dropout(opt.dropout)
        self.classifier = nn.Linear(opt.hidden_size + opt.hidden_dim_2, opt.num_labels)
        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    def forward(self,input_ids, bbox, attention_mask, pixel_values, labels, graph_vect):
        outputs = self.layoutlm(input_ids = input_ids, bbox = bbox, attention_mask = attention_mask, pixel_values = pixel_values)
        
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        hidden_state = outputs[0]   # hidden state has a shape of (batch_size, max_seq_len, dim)
        # only take the text part (the latter might be vision encoding)
        sequence_output = hidden_state[:, :seq_length]    # take (batch_size, seq, dim)
        
        full_sequence = torch.cat((sequence_output,graph_vect), -1) 
        sequence_output = self.dropout(full_sequence)

        # sequence classification will take the CLS vec, i.e.,
        # sequence_output = outputs[0][:, 0, :]

        # calculate label and loss
        logits = self.classifier(sequence_output)   # label prob distribution 

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss,logits
        # return outputs
