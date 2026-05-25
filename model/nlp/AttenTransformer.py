# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function

class AttentionTransformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(AttentionTransformer, self).__init__()

        self.input_dim = 768
        self.hidden_dim = config.getint('model', 'hidden_dim')
        
        try:
            self.num_heads = config.getint('model', 'num_heads')
        except Exception:
            self.num_heads = 8 # default fallback
            
        self.dropout_fc = config.getfloat('model', 'dropout_fc')
        self.num_layers = config.getint("model", 'num_layers')
        self.output_dim = config.getint("model", "output_dim")

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout_fc,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.fc_f = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_fc)
        self.weight = self.init_weight(config, gpu_list)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_weight(self, config, gpu_list):
        try:
            label_weight = config.getfloat('model', 'label_weight')
        except Exception:
            return None
        weight_lst = torch.ones(self.output_dim)
        weight_lst[-1] = label_weight
        if torch.cuda.is_available() and len(gpu_list) > 0:
            weight_lst = weight_lst.cuda()
        return weight_lst

    def init_multi_gpu(self, device, config, *args, **params):
        self.transformer = nn.DataParallel(self.transformer, device_ids=device)
        self.fc_f = nn.DataParallel(self.fc_f, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input']
        
        out = self.transformer(x)
        
        pooled_out = out.mean(dim=1)
        
        pooled_out = self.dropout(pooled_out)
        y = self.fc_f(pooled_out)
        y = y.view(y.size()[0], -1)

        if 'label' in data.keys():
            label = data['label']
            loss = self.criterion(y, label.view(-1))
            acc_result = self.accuracy_function(y, label, config, acc_result)
            if mode == 'valid' or mode == 'train':
                output = []
                y = y.cpu().detach().numpy().tolist()
                for i, guid in enumerate(data['guid']):
                    output.append([guid, label[i], y[i]])
                return {"loss": loss, "acc_result": acc_result, "output": output}
            elif mode == 'test':
                output = []
                y = y.cpu().detach().numpy().tolist()
                for i, guid in enumerate(data['guid']):
                    output.append([guid, y[i]])
                return {"output": output}
            return {"loss": loss, "acc_result": acc_result}
        else:
            output = []
            y = y.cpu().detach().numpy().tolist()
            for i, guid in enumerate(data['guid']):
                output.append([guid, y[i]])
            return {"output": output}
