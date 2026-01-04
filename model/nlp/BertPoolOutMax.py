# -*- coding: utf-8 -*-
__author__ = 'yshao'


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

class BertPoolOutMax(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPoolOutMax, self).__init__()
        self.max_para_c = config.getint('model', 'max_para_c')
        self.max_para_q = config.getint('model', 'max_para_q')
        self.step = config.getint('model', 'step')
        self.max_len = config.getint("data", "max_seq_length")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        # self.maxpool = nn.MaxPool1d(kernel_size=self.max_para_c)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, self.max_para_c), return_indices=True)
        self.provenance_service = None

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def set_provenance_service(self, provenance_service):
        self.provenance_service = provenance_service

    def forward(self, data, config, gpu_list, acc_result, mode, epoch=None):
        input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
        with torch.no_grad():
            output = []
            for k in tqdm(range(input_ids.size()[0]), desc="Pairs: ", leave=False):
                q_lst = []
                all_max_out = []
                all_selected_c_indices = []
                all_original_lst = []
                for i in tqdm(range(0, self.max_para_q, self.step), desc="Interactions: ", leave=False):
                    # print(input_ids[k, i:i+self.step].view(-1, self.max_len).size())
                    _, lst = self.bert(input_ids[k, i:i+self.step].view(-1, self.max_len),
                                       token_type_ids=token_type_ids[k, i:i+self.step].view(-1, self.max_len),
                                       attention_mask=attention_mask[k, i:i+self.step].view(-1, self.max_len))
                    # print('before view', lst.size())
                    lst = lst.view(self.step, self.max_para_c, -1)
                    # print('after view', lst.size())
                    lst = lst.permute(2, 0, 1)
                    # print('after permute', lst.size())
                    lst = lst.unsqueeze(0)
                    # print('after unsquezze', lst.size()) -> torch.Size([1, 768, 3, 40]) x, embedding, step(q), max_para_c
                    max_out, max_indices = self.maxpool(lst)
                    # print('after maxpool', max_out.size())
                    max_out = max_out.squeeze()
                    # print('after squeeze', max_out.size())
                    
                    max_indices = max_indices.squeeze()
                    # Convert flattened indices to c_para indices
                    selected_c_indices = max_indices % self.max_para_c  # Shape: [768, step]
                    
                    max_out = max_out.transpose(0, 1) 
                    # print('after transpose', max_out.size()) -> torch.Size([3, 768]) step(q), embedding (max_pooling over c)
                    selected_c_indices = selected_c_indices.transpose(0, 1)  # [step, 768]

                    # Collect data for concatenation
                    all_max_out.append(max_out)
                    all_selected_c_indices.append(selected_c_indices)
                    all_original_lst.append(lst.squeeze(0).permute(1, 2, 0))  # [step, max_para_c, 768]
                    
                    q_lst.extend(max_out.cpu().tolist())
                    #input('continue?')
                # print(len(q_lst))
                #exit()
                assert (len(q_lst) == self.max_para_q)
                output.append([data['guid'][k], q_lst])
                             # Concatenate all steps and store provenance once per sample
                if self.provenance_service is not None:
                    provenance_data = {
                        'guid': data['guid'][k],
                        'max_out': torch.cat(all_max_out, dim=0).cpu().tolist(),  # [max_para_q, 768]
                        'selected_c_indices': torch.cat(all_selected_c_indices, dim=0).cpu().tolist(),  # [max_para_q, 768]
                        'original_lst': torch.cat(all_original_lst, dim=0).cpu().tolist()  # [max_para_q, max_para_c, 768]
                    }
                    self.provenance_service.set_bert_scores_calculation(provenance_data, epoch)
                    
            return {"output": output}
        
