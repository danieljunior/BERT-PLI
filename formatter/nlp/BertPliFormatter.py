# -*- coding: utf-8 -*-
__author__ = 'yshao'

import json
import torch
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter
from .bert_feature_tool import example_item_to_feature


class BertPliFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

    def process(self, data, config, mode, *args, **params):
        return data

