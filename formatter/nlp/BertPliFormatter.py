# -*- coding: utf-8 -*-
from formatter.Basic import BasicFormatter

class BertPliFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

    def process(self, data, config, mode, *args, **params):
        return data
