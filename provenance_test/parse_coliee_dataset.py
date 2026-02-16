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


parse_coliee_dataset = Task(2 , tag, "parse_coliee_dataset")
coliee_dataset = DataSet("coliee_dataset", [Element(["/path/to/coliee/path","train"])])
parse_coliee_dataset.add_dataset(coliee_dataset)
parse_coliee_dataset.begin()
coliee_parsed_dataset = DataSet("coliee_parsed_dataset", [Element(["/path/to/coliee_parsed/path","train"])])
parse_coliee_dataset.add_dataset(coliee_parsed_dataset)
parse_coliee_dataset.end()

parse_coliee_dataset_file = '/app/test/coliee_parse.bin'
with open(parse_coliee_dataset_file, 'wb') as f:
    pickle.dump(parse_coliee_dataset, f)

dependencies['parse_coliee_dataset'] = parse_coliee_dataset_file

with open('/app/test/dependencies.bin', 'wb') as f:
    pickle.dump(dependencies, f)
