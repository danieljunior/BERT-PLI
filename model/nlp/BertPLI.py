import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config_parser import create_config

from model.nlp.BertPoolOutMax import BertPoolOutMax
from model.nlp.AttenRNN import AttentionRNN
from formatter.nlp.BertDocParaFormatter import BertDocParaFormatter

logger = logging.getLogger(__name__)

class BertPLI(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPLI, self).__init__()
        self.poolout_max = BertPoolOutMax(self.poolout_config(config), 
                                            gpu_list, *args, **params)        
        self.attention_rnn = AttentionRNN(self.attention_rnn_config(config),
                                        gpu_list, *args, **params)

    def forward(self, data, config, gpu_list, acc_result, mode):
        data = self.tokenize_data(data, config, gpu_list, acc_result, mode)
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
    
    def set_tokenizer_formatter(self, mode, config, *args, **params):
        self.tokenizer_formatter = BertDocParaFormatter(config, mode, *args, **params)

    def tokenize_data(self, data, config, gpu_list, acc_result, mode):
        data = self.tokenizer_formatter.process(data, config, gpu_list, acc_result, mode)
        
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])
        
        return data

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