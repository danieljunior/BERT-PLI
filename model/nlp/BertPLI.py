import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config_parser import create_config

from formatter.nlp.BertDocParaFormatter import BertDocParaFormatter
from model.nlp.BertPoolOutMax import BertPoolOutMax
from model.nlp.AttenRNN import AttentionRNN
from model.nlp.BypassSelection import BypassSelection
from model.nlp.SumySelection import SumySelection
from model.nlp.LearnableSequenceSelector import LearnableSequenceSelector

logger = logging.getLogger(__name__)


class BertPLI(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertPLI, self).__init__()
        self.poolout_max = BertPoolOutMax(
            self.poolout_config(config), gpu_list, *args, **params
        )
        self.attention_rnn = AttentionRNN(
            self.attention_rnn_config(config), gpu_list, *args, **params
        )
        self.selection_mode = None
        self.provenance_service = None

    def forward(self, data, config, gpu_list, acc_result, mode, epoch=None):
        data = self.select_segments(data, epoch)
        data = self.tokenize_data(data, config, gpu_list, acc_result, mode)
        #TODO capture bert_scores_calculation and max_pooling provenance
        poolout = self.poolout_max(
            data, self.poolout_config(config), gpu_list, acc_result, mode
        )
        poolout = {guid: result for guid, result in poolout["output"]}
        labels = data["label"] if mode != "test" else []
        rnn_input = self.poolout_to_rnn(poolout, labels, mode=mode)
        result = self.attention_rnn(
            rnn_input, self.attention_rnn_config(config), gpu_list, acc_result, mode
        )
        return result

    def init_multi_gpu(self, device, config, *args, **params):
        self.poolout_max.init_multi_gpu(device, config, *args, **params)
        # self.attention_rnn.init_multi_gpu(device, config, *args, **params)

    def set_provenance_service(self, provenance_service):
        self.provenance_service = provenance_service

    def set_tokenizer_formatter(self, mode, config, *args, **params):
        self.tokenizer_formatter = BertDocParaFormatter(config, mode, *args, **params)

    def set_selection_layer(self, selection_mode):
        self.selection_mode = selection_mode
        if selection_mode == "sumy":
            self.selection_layer = SumySelection()
        elif selection_mode == "learnable":
            selection_layer_ = LearnableSequenceSelector(
                embed_dim=768, num_heads=1, num_to_select=20
            )
            self.add_module("selection_layer", selection_layer_)
        else:
            self.selection_layer = BypassSelection()

    def poolout_config(self, config):
        return create_config(config.get("poolout", "config_file"))

    def attention_rnn_config(self, config):
        return create_config(config.get("attention_rnn", "config_file"))

    def tokenize_data(self, data, config, gpu_list, acc_result, mode):
        data = self.tokenizer_formatter.process(
            data, config, gpu_list, acc_result, mode
        )

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        return data

    def select_segments(self, data, epoch):
        if self.selection_mode == "learnable":
            data = self._learnable_selection(data)
        else:
            data = self.selection_layer.forward(data)
        #TODO test learnable selection provenance
        self.provenance_service.set_get_relevant_segments_task(
            data, criteria=self.selection_mode, epoch=epoch
        )
        return data

    def _learnable_selection(self, data):
        c_paras = [d["c_paras"] for d in data]
        q_paras = [d["q_paras"] for d in data]
        selected_c_idx, selected_c_paras = self._select_segments_one_text(c_paras)
        for row, paras, idxs in zip(data, selected_c_paras, selected_c_idx):
            row["c_paras"] = paras
            row["c_selected_indices"] = idxs

        selected_q_idx, selected_q_paras = self._select_segments_one_text(q_paras)
        for row, paras, idxs in zip(data, selected_q_paras, selected_q_idx):
            row["q_paras"] = paras
            row["q_selected_indices"] = idxs
    
    def _select_segments_one_text(self, paras):
        selected_sequences, selection_indices, scores = self.selection_layer.forward(
            paras
        )
        # convert tensor indices to plain python lists if needed
        if isinstance(selection_indices, torch.Tensor):
            selection_indices = selection_indices.cpu().detach().tolist()

        # selection_indices is now a list (per-doc) of indices to keep
        selected_paras = []
        selected_indices = []
        for paras, idxs in zip(paras, selection_indices):
            # ensure idxs is a list
            if isinstance(idxs, int):
                idxs = [idxs]
            # eu faço padding, então pode retornar índices fora do alcance
            tmp_selected_indices = [i for i in idxs if i < len(paras)]
            selected = [paras[i] for i in tmp_selected_indices]
            selected_indices.append(tmp_selected_indices)
            selected_paras.append(selected)
        return selected_indices, selected_paras

    def poolout_to_rnn(self, data, labels, mode="train"):
        inputs = []
        guids = []

        for i, (guid, emb_mtx) in enumerate(data.items()):
            inputs.append(emb_mtx)
            guids.append(guid)

        inputs = torch.tensor(inputs)

        if mode != "test":
            return {"guid": guids, "input": inputs.cuda(), "label": labels.cuda()}
        else:
            return {"guid": guids, "input": inputs.cuda()}
