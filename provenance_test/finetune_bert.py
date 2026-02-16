from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
import pickle

tag = 'test_dataflow'

dependencies = {}
try:
    with open('/app/test/dependencies.bin', 'rb') as f:
        dependencies = pickle.load(f)
except FileNotFoundError:
    pass


finetune_bert = Task(1 , tag, "finetune_bert")
bert_base = DataSet("bert_base", [Element(["/path/to/bert_base/checkpoint"])])
finetune_bert.add_dataset(bert_base)
entailment_config = DataSet("entailment_config", [Element(["/path/to/entailment/config"])])
finetune_bert.add_dataset(entailment_config)
finetune_bert.begin()
finetuned_bert = DataSet("finetuned_bert_model", [Element(["/path/to/finetuned_bert/last_checkpoint"])])
finetune_bert.add_dataset(finetuned_bert)
finetune_bert.end()

finetune_bert_file = '/app/test/finetune_bert.bin'
with open(finetune_bert_file, 'wb') as f:
    pickle.dump(finetune_bert, f)

dependencies['finetune_bert'] = finetune_bert_file

with open('/app/test/dependencies.bin', 'wb') as f:
    pickle.dump(dependencies, f)
