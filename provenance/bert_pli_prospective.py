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
    tf1_output.set_type(SetType.INPUT)
    tf1_output.dependency = tf1._tag
    tf2_output = Set("splitted_doc_pair_dataset", SetType.OUTPUT, [
        Attribute("filepath", AttributeType.FILE),
        Attribute("type", AttributeType.TEXT),
        Attribute("count", AttributeType.NUMERIC),
        Attribute("split_method", AttributeType.TEXT),
        Attribute("split_level", AttributeType.TEXT)
    ])
    tf2.set_sets([tf1_output, tf2_output])
    df.add_transformation(tf2)
    
    tf2_output.set_type(SetType.INPUT)
    tf2_output.dependency = tf2._tag
    # Transformation 3: get_doc1_example
    tf3 = Transformation("get_doc1_example")
    tf3_output = Set("doc1_segment", SetType.OUTPUT, [
        Attribute("file_id", AttributeType.TEXT),
        Attribute("filepath", AttributeType.FILE),
        Attribute("text", AttributeType.TEXT),
        Attribute("idx", AttributeType.NUMERIC)
    ])
    tf3.set_sets([tf2_output, tf3_output])
    df.add_transformation(tf3)

    # Transformation 4: get_doc2_example
    tf4 = Transformation("get_doc2_example")
    tf4_output = Set("doc2_segment", SetType.OUTPUT, [
        Attribute("file_id", AttributeType.TEXT),
        Attribute("filepath", AttributeType.FILE),
        Attribute("text", AttributeType.TEXT),
        Attribute("idx", AttributeType.NUMERIC)
    ])
    tf4.set_sets([tf2_output, tf4_output])
    df.add_transformation(tf4)

    # Transformation 5: get_label_example
    tf5 = Transformation("get_label_example")
    tf5_output = Set("label", SetType.OUTPUT, [
        Attribute("value", AttributeType.NUMERIC)
    ])
    tf5.set_sets([tf2_output, tf5_output])
    df.add_transformation(tf5)
    
    # Transformation 6: doc1_relevant_segments_selection
    tf6 = Transformation("doc1_relevant_segments_selection")
    tf3_output.set_type(SetType.INPUT)
    tf3_output.dependency = tf3._tag
    tf6_output = Set("doc1_relevant_segment", SetType.OUTPUT, [
        Attribute("file_id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("idx", AttributeType.NUMERIC),
        Attribute("criteria", AttributeType.TEXT),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf6.set_sets([tf3_output, tf6_output])
    df.add_transformation(tf6)
    
    # Transformation 7: doc2_relevant_segments_selection
    tf7 = Transformation("doc2_relevant_segments_selection")
    tf4_output.set_type(SetType.INPUT)
    tf4_output.dependency = tf4._tag
    tf7_output = Set("doc2_relevant_segment", SetType.OUTPUT, [
        Attribute("file_id", AttributeType.TEXT),
        Attribute("text", AttributeType.TEXT),
        Attribute("idx", AttributeType.NUMERIC),
        Attribute("criteria", AttributeType.TEXT),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf7.set_sets([tf4_output, tf7_output])
    df.add_transformation(tf7)
    
    # Transformation 8: bert_scores_calculation
    tf8 = Transformation("bert_scores_calculation")
    tf6_output.set_type(SetType.INPUT)
    tf6_output.dependency = tf6._tag
    tf7_output.set_type(SetType.INPUT)
    tf7_output.dependency = tf7._tag
    tf8_output = Set("interaction_map", SetType.OUTPUT, [
        Attribute("row", AttributeType.NUMERIC),
        Attribute("scores", AttributeType.TEXT),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf8.set_sets([tf6_output, tf7_output, tf8_output])
    df.add_transformation(tf8)
    
    # Transformation 9: max_pooling
    tf9 = Transformation("max_pooling")
    tf8_output.set_type(SetType.INPUT)
    tf8_output.dependency = tf8._tag
    tf9_output = Set("feature_vector", SetType.OUTPUT, [
        Attribute("idx", AttributeType.NUMERIC),
        Attribute("value", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf9.set_sets([tf8_output, tf9_output])
    df.add_transformation(tf9)
    
    # Transformation 10: classification
    tf10 = Transformation("classification")
    tf9_output.set_type(SetType.INPUT)
    tf9_output.dependency = tf9._tag
    tf10_output = Set("output", SetType.OUTPUT, [
        Attribute("predicted_label", AttributeType.NUMERIC),
        Attribute("loss_metric", AttributeType.TEXT),
        Attribute("loss_value", AttributeType.NUMERIC),
        Attribute("epoch", AttributeType.NUMERIC)
    ])
    tf10.set_sets([tf9_output, tf10_output])
    df.add_transformation(tf10)
    
    # Transformation 11: evaluating
    tf11 = Transformation("evaluating")
    tf10_output.set_type(SetType.INPUT)
    tf10_output.dependency = tf10._tag
    tf5_output.set_type(SetType.INPUT)
    tf5_output.dependency = tf5._tag
    tf11_output = Set("metric", SetType.OUTPUT, [
        Attribute("type", AttributeType.TEXT),
        Attribute("value", AttributeType.NUMERIC)
    ])
    tf11.set_sets([tf10_output, tf5_output, tf11_output])
    df.add_transformation(tf11)
    
    return df

if __name__ == "__main__":
    df = create_bert_pli_dataflow("bert-pli")
    df.save()
    
    print(f"Dataflow '{df._tag}' created and saved successfully!")
