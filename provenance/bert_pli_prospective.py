from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
import os

os.environ['DFA_URL'] = 'http://dfanalyzer:22000/'   

def create_bert_pli_dataflow(dataflow_tag="bert-pli3"):
    """Create BERT-PLI dataflow with prospective provenance"""
    
    df = Dataflow(dataflow_tag)
    
    # Transformation 1: docs_pairs_generation
    tf1 = Transformation("docs_pairs_generation")
    tf1_input = Set("coliee_file_input", SetType.INPUT, [
        Attribute("filepath", AttributeType.FILE),
        Attribute("split_type", AttributeType.TEXT)
    ])
    tf1_output = Set("doc_pair_dataset", SetType.OUTPUT, [
        Attribute("filepath", AttributeType.FILE),
        Attribute("split_type", AttributeType.TEXT),
        Attribute("count", AttributeType.NUMERIC)
    ])
    tf1.set_sets([tf1_input, tf1_output])
    df.add_transformation(tf1)
    
    # Transformation 2: docs_texts_splitting
    tf2 = Transformation("docs_texts_splitting")
    tf2_input = Set("doc_pair_dataset", SetType.INPUT, [
        Attribute("filepath", AttributeType.FILE),
        Attribute("type", AttributeType.TEXT),
        Attribute("count", AttributeType.NUMERIC)
    ])
    tf2_input.dependency = tf1._tag
    tf2_output = Set("splitted_doc_pair_dataset", SetType.OUTPUT, [
        Attribute("filepath", AttributeType.FILE),
        Attribute("type", AttributeType.TEXT),
        Attribute("count", AttributeType.NUMERIC),
        Attribute("split_method", AttributeType.TEXT),
        Attribute("split_level", AttributeType.TEXT)
    ])
    tf2.set_sets([tf2_input, tf2_output])
    df.add_transformation(tf2)
    
    # Transformation 3: get_example
    tf3 = Transformation("get_example")
    tf3_input = Set("splitted_doc_pair_dataset", SetType.INPUT, [
        Attribute("filepath", AttributeType.FILE),
        Attribute("type", AttributeType.TEXT),
        Attribute("count", AttributeType.NUMERIC),
        Attribute("split_method", AttributeType.TEXT),
        Attribute("split_level", AttributeType.TEXT)
    ])
    tf3_input.dependency = tf2._tag
    tf3_output1 = Set("doc1_segment", SetType.OUTPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("filepath", AttributeType.FILE),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC)
    ])
    tf3_output2 = Set("doc2_segment", SetType.OUTPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("filepath", AttributeType.FILE),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC)
    ])
    tf3_output3 = Set("label", SetType.OUTPUT, [
        Attribute("value", AttributeType.NUMERIC)
    ])
    tf3.set_sets([tf3_input, tf3_output1, tf3_output2, tf3_output3])
    df.add_transformation(tf3)
    
    # Transformation 4: doc1_relevant_segments_selection
    tf4 = Transformation("doc1_relevant_segments_selection")
    tf4_input = Set("doc1_segment", SetType.INPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("filepath", AttributeType.FILE),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC)
    ])
    tf4_input.dependency = tf3._tag
    tf4_output = Set("doc1_relevant_segment", SetType.OUTPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf4.set_sets([tf4_input, tf4_output])
    df.add_transformation(tf4)
    
    # Transformation 5: doc2_relevant_segments_selection
    tf5 = Transformation("doc2_relevant_segments_selection")
    tf5_input = Set("doc2_segment", SetType.INPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf5_input.dependency = tf3._tag
    tf5_output = Set("doc2_relevant_segment", SetType.OUTPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf5.set_sets([tf5_input, tf5_output])
    df.add_transformation(tf5)
    
    # Transformation 6: bert_scores_calculation
    tf6 = Transformation("bert_scores_calculation")
    tf6_input1 = Set("doc1_relevant_segment", SetType.INPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf6_input1.dependency = tf4._tag
    tf6_input2 = Set("doc2_relevant_segment", SetType.INPUT, [
        Attribute("id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("seq", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf6_input2.dependency = tf5._tag
    tf6_output = Set("interaction_map", SetType.OUTPUT, [
        Attribute("row", AttributeType.NUMERIC),
        Attribute("scores", AttributeType.TEXT),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf6.set_sets([tf6_input1, tf6_input2, tf6_output])
    df.add_transformation(tf6)
    
    # Transformation 7: max_pooling
    tf7 = Transformation("max_pooling")
    tf7_input = Set("interaction_map", SetType.INPUT, [
        Attribute("row", AttributeType.NUMERIC),
        Attribute("scores", AttributeType.TEXT),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf7_input.dependency = tf6._tag
    tf7_output = Set("feature_vector", SetType.OUTPUT, [
        Attribute("cell", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf7.set_sets([tf7_input, tf7_output])
    df.add_transformation(tf7)
    
    # Transformation 8: classification
    tf8 = Transformation("classification")
    tf8_input = Set("feature_vector", SetType.INPUT, [
        Attribute("cell", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf8_input.dependency = tf7._tag
    tf8_output = Set("output", SetType.OUTPUT, [
        Attribute("predicted_label", AttributeType.NUMERIC),
        Attribute("loss", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf8.set_sets([tf8_input, tf8_output])
    df.add_transformation(tf8)
    
    # Transformation 9: evaluating
    tf9 = Transformation("evaluating")
    tf9_input1 = Set("output", SetType.INPUT, [
        Attribute("predicted_label", AttributeType.NUMERIC),
        Attribute("loss", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf9_input1.dependency = tf8._tag
    tf9_input2 = Set("label", SetType.INPUT, [
        Attribute("value", AttributeType.NUMERIC)
    ])
    tf9_input2.dependency = tf3._tag
    tf9_output = Set("metric", SetType.OUTPUT, [
        Attribute("type", AttributeType.TEXT),
        Attribute("value", AttributeType.NUMERIC)
    ])
    tf9.set_sets([tf9_input1, tf9_input2, tf9_output])
    df.add_transformation(tf9)
    
    return df

if __name__ == "__main__":
    df = create_bert_pli_dataflow("bert-pli4")
    df.save()
    
    print(f"Dataflow '{df._tag}' created and saved successfully!")
