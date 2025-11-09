import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from config_parser import create_config

from model.nlp.BertPoolOutMax import BertPoolOutMax
from model.nlp.AttenRNN import AttentionRNN

logger = logging.getLogger(__name__)

class BertPLI(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPLI, self).__init__()
        self.poolout_max = BertPoolOutMax(self.poolout_config(config), 
                                            gpu_list, *args, **params)        
        self.attention_rnn = AttentionRNN(self.attention_rnn_config(config),
                                        gpu_list, *args, **params)

    def forward(self, data, config, gpu_list, acc_result, mode):
        poolout = self.poolout_max(data, self.poolout_config(config),
                                    gpu_list, acc_result, mode)
        poolout = {guid: result for guid, result in poolout['output']}
        labels = data['label'] if mode != 'test' else []
        rnn_input = self.poolout_to_rnn(poolout, labels, mode=mode)
        result = self.attention_rnn(rnn_input, self.attention_rnn_config(config), gpu_list, acc_result, mode)
        return result

    def init_multi_gpu(self, device, config, *args, **params):
        self.poolout_max.init_multi_gpu(device, config, *args, **params)
        # self.attention_rnn.init_multi_gpu(device, config, *args, **params)

    def poolout_config(self, config):
        return create_config(config.get('poolout', 'config_file'))
    
    def attention_rnn_config(self, config):
        return create_config(config.get('attention_rnn', 'config_file'))
    
    def poolout_to_rnn(self, data, labels, mode="train"):
        inputs = []
        guids = []

        for i, (guid, emb_mtx) in enumerate(data.items()):
            inputs.append(emb_mtx)
            guids.append(guid)

        inputs = torch.tensor(inputs)

        if mode != 'test':
            return {'guid': guids, 'input': inputs.cuda(), 'label': labels.cuda()}
        else:
            return {'guid': guids, 'input': inputs.cuda()}