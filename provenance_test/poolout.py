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

# load dependencies
with open(dependencies['parse_coliee_dataset'], 'rb') as f:
    coliee_parse = pickle.load(f)

with open(dependencies['finetune_bert'], 'rb') as f:
    finetune_bert = pickle.load(f)

poolout = Task(3 , tag, "poolout", dependency=[coliee_parse, finetune_bert])
coliee_dataset = DataSet("poolout_config", [Element(["blablabla"])])
poolout.add_dataset(coliee_dataset)
poolout.begin()
poolout_data = DataSet("poolout_data", [Element(["poolout_file","selected_sentences_file"])])
poolout.add_dataset(poolout_data)
poolout.end()

poolout_file = '/app/test/poolout.bin'
with open(poolout_file, 'wb') as f:
    pickle.dump(poolout, f)

dependencies['poolout'] = poolout

with open('/app/test/dependencies.bin', 'wb') as f:
    pickle.dump(dependencies, f)