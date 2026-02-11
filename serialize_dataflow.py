from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from time import sleep
import pickle

tag = 'test_dataflow'
df = Dataflow(tag)
    
# Transformation 1: docs_pairs_generation
tf1 = Transformation("parse_coliee_dataset")
coliee_dataset = Set("coliee_dataset", SetType.INPUT, [
    Attribute("filepath", AttributeType.FILE),
    Attribute("split_type", AttributeType.TEXT)
])
coliee_parsed_dataset = Set("coliee_parsed_dataset", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE),
    Attribute("split_type", AttributeType.TEXT)
])
tf1.set_sets([coliee_dataset, coliee_parsed_dataset])
df.add_transformation(tf1)

tf2 = Transformation("poolout")
coliee_parsed_dataset.set_type(SetType.INPUT)
coliee_parsed_dataset.dependency = tf1._tag

# finetuned_bert_model = Set("finetuned_bert_model", SetType.INPUT, [
#     Attribute("checkpoint_path", AttributeType.FILE)])

poolout_data = Set("poolout_data", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE)
])
# tf2.set_sets([coliee_parsed_dataset, finetuned_bert_model, poolout_data])
tf2.set_sets([coliee_parsed_dataset, poolout_data])
df.add_transformation(tf2)

df.save()

####################### Retrospective

t1 = Task(1 , tag, "parse_coliee_dataset")
t1_input = DataSet("coliee_dataset", [Element(["/path/to/coliee/data", "train"])])
t1.add_dataset(t1_input)

t1.begin()
sleep(5)  # Simulate some processing time
t1_output = DataSet("coliee_parsed_dataset", [Element(["/path/to/coliee/parsed_data", "train"])])
t1.add_dataset(t1_output)
t1.end()

print("Serialize Task:", t1)
# Serialize the task to a file
with open('tf1.pkl', 'wb') as f:
    pickle.dump(t1, f)

print("Serialize Dataset:", t1_output)
# Serialize the task to a file
with open('t1_output.pkl', 'wb') as f:
    pickle.dump(t1_output, f)

print("Deserialize Task: T1")
# Deserialize the task from the file
with open('tf1.pkl', 'rb') as f:
    t1_loaded = pickle.load(f)

print("Deserialize Dataset: T1output")
# Deserialize the task from the file
with open('t1_output.pkl', 'rb') as f:
    t1_output_loaded = pickle.load(f)

t2 = Task(2 , tag, "poolout", dependency=t1_loaded)
t2.begin()

t2_output = DataSet(
    "poolout_data",
    [Element(['poolout.json'])],
)
t2.add_dataset(t2_output)
t2.end()

print("Finish")