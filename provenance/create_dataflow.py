from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType

tag = 'test_dataflow'
df = Dataflow(tag)

tf_finetune_bert = Transformation("finetune_bert")
dt_bert_base = Set("bert_base", SetType.INPUT, [
    Attribute("checkpoint", AttributeType.FILE)])
dt_entailment_config = Set("entailment_config", SetType.INPUT, [
    Attribute("config", AttributeType.TEXT)])
dt_finetuned_bert = Set("finetuned_bert_model", SetType.OUTPUT, [
    Attribute("epoch", AttributeType.TEXT),
    Attribute("checkpoint", AttributeType.FILE),
    ])
tf_finetune_bert.set_sets([dt_bert_base, dt_entailment_config, dt_finetuned_bert])
df.add_transformation(tf_finetune_bert)

tf_parse_coliee = Transformation("parse_coliee_dataset")
dt_coliee_dataset = Set("coliee_dataset", SetType.INPUT, [
    Attribute("filepath", AttributeType.FILE),
    Attribute("split_type", AttributeType.TEXT)])
dt_coliee_parsed_dataset = Set("coliee_parsed_dataset", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE),
    Attribute("split_type", AttributeType.TEXT)])
tf_parse_coliee.set_sets([dt_coliee_dataset, dt_coliee_parsed_dataset])
df.add_transformation(tf_parse_coliee)

tf_poolout = Transformation("poolout")
dt_coliee_parsed_dataset.set_type(SetType.INPUT)
dt_coliee_parsed_dataset.dependency = tf_parse_coliee._tag
dt_finetuned_bert.set_type(SetType.INPUT)
dt_finetuned_bert.dependency = tf_finetune_bert._tag
dt_poolout_config = Set("poolout_config", SetType.INPUT, [
    Attribute("config", AttributeType.TEXT)])
dt_poolout_data = Set("poolout_data", SetType.OUTPUT, [
    Attribute("epoch", AttributeType.TEXT),
    Attribute("poolout_filepath", AttributeType.FILE),
    Attribute("selected_sentences_filepath", AttributeType.FILE)])
tf_poolout.set_sets([dt_finetuned_bert, dt_coliee_parsed_dataset, dt_poolout_config, 
                     dt_poolout_data])
df.add_transformation(tf_poolout)

tf_parse_poolout = Transformation("parse_poolout")
dt_poolout_data.set_type(SetType.INPUT)
dt_poolout_data.dependency = tf_poolout._tag
dt_parsed_poolout_data = Set("parsed_poolout_data", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE)])
tf_parse_poolout.set_sets([dt_poolout_data, dt_parsed_poolout_data])
df.add_transformation(tf_parse_poolout)

tf_train_classifier = Transformation("train_classifier")
dt_parsed_poolout_data.set_type(SetType.INPUT)
dt_parsed_poolout_data.dependency = tf_parse_poolout._tag
dt_classifier_config = Set("classifier_config", SetType.INPUT, [
    Attribute("config", AttributeType.TEXT)])
dt_classifier_model = Set("classifier_model", SetType.OUTPUT, [
    Attribute("epoch", AttributeType.TEXT),
    Attribute("filepath", AttributeType.FILE),
    Attribute("validation_metrics_filepath", AttributeType.FILE)])
tf_train_classifier.set_sets([dt_parsed_poolout_data, dt_classifier_config, dt_classifier_model])
df.add_transformation(tf_train_classifier)

tf_test = Transformation("test_classifier")
dt_classifier_model.set_type(SetType.INPUT)
dt_classifier_model.dependency = tf_train_classifier._tag
dt_test_config = Set("test_config", SetType.INPUT, [
    Attribute("config", AttributeType.TEXT)])
dt_test_results = Set("test_results", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE)])
tf_test.set_sets([dt_classifier_model, dt_test_config, dt_coliee_parsed_dataset, dt_test_results])
df.add_transformation(tf_test)

tf_parse_results = Transformation("parse_results")
dt_test_results.set_type(SetType.INPUT)
dt_test_results.dependency = tf_test._tag
dt_parsed_test_results = Set("parsed_test_results", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE)])
tf_parse_results.set_sets([dt_test_results, dt_parsed_test_results])
df.add_transformation(tf_parse_results)

tf_calculate_metrics = Transformation("calculate_metrics")
dt_parsed_test_results.set_type(SetType.INPUT)
dt_parsed_test_results.dependency = tf_parse_results._tag
dt_true_labels = Set("true_labels", SetType.INPUT, [
    Attribute("filepath", AttributeType.FILE)])
dt_metrics = Set("metrics", SetType.OUTPUT, [
    Attribute("filepath", AttributeType.FILE)])
tf_calculate_metrics.set_sets([dt_parsed_test_results, dt_true_labels, dt_metrics])
df.add_transformation(tf_calculate_metrics)

df.save()

####################### Retrospective

# t1 = Task(1 , tag, "parse_coliee_dataset")
# t1_input = DataSet("coliee_dataset", [Element(["/path/to/coliee/data", "train"])])
# t1.add_dataset(t1_input)

# t1.begin()
# sleep(5)  # Simulate some processing time
# t1_output = DataSet("coliee_parsed_dataset", [Element(["/path/to/coliee/parsed_data", "train"])])
# t1.add_dataset(t1_output)
# t1.end()

# print("Serialize Task:", t1)
# # Serialize the task to a file
# with open('tf1.pkl', 'wb') as f:
#     pickle.dump(t1, f)

# print("Serialize Dataset:", t1_output)
# # Serialize the task to a file
# with open('t1_output.pkl', 'wb') as f:
#     pickle.dump(t1_output, f)

# print("Deserialize Task: T1")
# # Deserialize the task from the file
# with open('tf1.pkl', 'rb') as f:
#     t1_loaded = pickle.load(f)

# print("Deserialize Dataset: T1output")
# # Deserialize the task from the file
# with open('t1_output.pkl', 'rb') as f:
#     t1_output_loaded = pickle.load(f)

# #TODO checar se após o load o grafo é populado com as dependências corretamente, ou seja, 
# # se t1_loaded tem a dependência do dataset t1_output_loaded
# import pdb; pdb.set_trace()
# t2 = Task(2 , tag, "poolout", dependency=t1_loaded)
# t2.begin()

# t2_output = DataSet(
#     "poolout_data",
#     [Element(['poolout.json'])],
# )
# t2.add_dataset(t2_output)
# t2.end()

# print("Finish")