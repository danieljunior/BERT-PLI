from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType


dataflow_tag = "test1"

df = Dataflow(dataflow_tag)
    
# Transformation 1: docs_pairs_generation
tf1 = Transformation("docs_pairs_generation")
tf1_input = Set("coliee_file_input", SetType.INPUT, [])
tf1_output = Set("doc_pair_dataset", SetType.OUTPUT, [])
tf1.set_sets([tf1_input, tf1_output])
df.add_transformation(tf1)

# Transformation 2: docs_texts_splitting
tf2 = Transformation("docs_texts_splitting")
tf2_input = Set("doc_pair_dataset", SetType.INPUT, [])
tf2_input.dependency = tf1._tag
tf2_output = Set("splitted_doc_pair_dataset", SetType.OUTPUT, [])
tf2.set_sets([tf2_input, tf2_output])
df.add_transformation(tf2)

# Transformation 3: get_example
tf3 = Transformation("get_example")
tf3_input = Set("splitted_doc_pair_dataset", SetType.INPUT, [])
tf3_input.dependency = tf2._tag
tf3_output1 = Set("doc1_segment", SetType.OUTPUT, [])
tf3_output2 = Set("doc2_segment", SetType.OUTPUT, [])
tf3_output3 = Set("label", SetType.OUTPUT, [])
tf3.set_sets([tf3_input, tf3_output1, tf3_output2, tf3_output3])
df.add_transformation(tf3)

# Transformation 4: doc1_relevant_segments_selection
tf4 = Transformation("doc1_relevant_segments_selection")
tf4_input = Set("doc1_segment", SetType.INPUT, [])
tf4_input.dependency = tf3_output1._tag
tf4_output = Set("doc1_relevant_segment", SetType.OUTPUT, [])
tf4.set_sets([tf4_input, tf4_output])
df.add_transformation(tf4)

# Transformation 5: doc2_relevant_segments_selection
tf5 = Transformation("doc2_relevant_segments_selection")
tf5_input = Set("doc2_segment", SetType.INPUT, [])
tf5_input.dependency = tf3_output2._tag
tf5_output = Set("doc2_relevant_segment", SetType.OUTPUT, [])
tf5.set_sets([tf5_input, tf5_output])
df.add_transformation(tf5)

df.save()