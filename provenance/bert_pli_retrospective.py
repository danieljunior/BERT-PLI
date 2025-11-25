from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
import time
import os

os.environ['DFA_URL'] = 'http://dfanalyzer:22000/'

from bert_pli_prospective import create_bert_pli_dataflow

# Create prospective provenance
df = create_bert_pli_dataflow(dataflow_tag="bert-pli-dataflow3")
df.save()

dataflow_tag = df._tag

# Retrospective provenance
# Task 1: docs_pairs_generation
t1 = Task(1, dataflow_tag, "docs_pairs_generation")
t1_input = DataSet("coliee_file_input", [
    Element(["/path/to/coliee/data", "train"])
])
t1.add_dataset(t1_input)
t1.begin()

# Simulated output data
filepath_pair = "/path/to/processed/doc_pairs.json"
split_type = "train"
count = 1000

t1_output = DataSet("doc_pair_dataset", [
    Element([filepath_pair, split_type, count])
])
t1.add_dataset(t1_output)
t1.end()
time.sleep(1)

# Task 2: docs_texts_splitting
t2 = Task(2, dataflow_tag, "docs_texts_splitting", dependency=t1)
t2.begin()

filepath_split = "/path/to/processed/doc_pairs_split.json"
split_method = "sentence"
split_level = "paragraph"
split_count = 1500

t2_output = DataSet("splitted_doc_pair_dataset", [
    Element([filepath_split, split_type, split_count, split_method, split_level])
])
t2.add_dataset(t2_output)
t2.end()
time.sleep(1)

# Task 3: get_example
t3 = Task(3, dataflow_tag, "get_example", dependency=t2)
t3.begin()

doc1_filepath = "/path/to/processed/doc1_segments.json"
doc2_filepath = "/path/to/processed/doc2_segments.json"
doc1_seq = 1500
doc2_seq = 1500
label_value = 1

t3_output1 = DataSet("doc1_segment", [
    Element([doc1_filepath, "sample doc1 text", doc1_seq])
])
t3_output2 = DataSet("doc2_segment", [
    Element([doc2_filepath, "sample doc2 text", doc2_seq])
])
t3_output3 = DataSet("label", [
    Element([label_value])
])

t3.add_dataset(t3_output1)
t3.add_dataset(t3_output2)
t3.add_dataset(t3_output3)
t3.end()
time.sleep(1)

# Task 4: doc1_relevant_segments_selection
t4 = Task(4, dataflow_tag, "doc1_relevant_segments_selection", dependency=t3)
t4.begin()

doc1_id = "doc1_001"
doc1_text = "relevant text from doc1"
doc1_seq_rel = 50
epoch_t4 = 1

t4_output = DataSet("doc1_relevant_segment", [
    Element([doc1_id, doc1_text, doc1_seq_rel, epoch_t4])
])
t4.add_dataset(t4_output)
t4.end()
time.sleep(1)

# Task 5: doc2_relevant_segments_selection
t5 = Task(5, dataflow_tag, "doc2_relevant_segments_selection", dependency=t3)
t5.begin()

doc2_id = "doc2_001"
doc2_text = "relevant text from doc2"
doc2_seq_rel = 50
epoch_t5 = 1

t5_output = DataSet("doc2_relevant_segment", [
    Element([doc2_id, doc2_text, doc2_seq_rel, epoch_t5])
])
t5.add_dataset(t5_output)
t5.end()
time.sleep(1)

# Task 6: bert_scores_calculation
t6 = Task(6, dataflow_tag, "bert_scores_calculation", dependency=[t4, t5])
t6.begin()

row_idx = 0
scores_str = "[0.95, 0.87, 0.92, 0.78]"
epoch_t6 = 1

t6_output = DataSet("interaction_map", [
    Element([row_idx, scores_str, epoch_t6])
])
t6.add_dataset(t6_output)
t6.end()
time.sleep(1)

# Task 7: max_pooling
t7 = Task(7, dataflow_tag, "max_pooling", dependency=t6)
t7.begin()

cell_idx = 0
epoch_t7 = 1

t7_output = DataSet("feature_vector", [
    Element([cell_idx, epoch_t7])
])
t7.add_dataset(t7_output)
t7.end()
time.sleep(1)

# Task 8: classification
t8 = Task(8, dataflow_tag, "classification", dependency=t7)
t8.begin()

predicted_label = 1
loss = 0.234
epoch_t8 = 1

t8_output = DataSet("output", [
    Element([predicted_label, loss, epoch_t8])
])
t8.add_dataset(t8_output)
t8.end()
time.sleep(1)

# Task 9: evaluating
t9 = Task(9, dataflow_tag, "evaluating", dependency=[t8, t3])
t9.begin()

metric_type = "accuracy"
metric_value = 0.92

t9_output = DataSet("metric", [
    Element([metric_type, metric_value])
])
t9.add_dataset(t9_output)
t9.end()

print(f"Retrospective provenance for dataflow '{dataflow_tag}' completed successfully!")
