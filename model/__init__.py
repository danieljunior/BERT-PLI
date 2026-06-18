from .nlp.BertPoint import BertPoint
from .nlp.BertPoolOutMax import BertPoolOutMax
from .nlp.AttenRNN import AttentionRNN
from .nlp.AttenTransformer import AttentionTransformer
from .nlp.BertPLI import BertPLI

model_list = {
    "BertPoint": BertPoint,
    "BertPoolOutMax": BertPoolOutMax,
    "AttenRNN": AttentionRNN,
    "AttenTransformer": AttentionTransformer,
    "BertPLI" : BertPLI
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
