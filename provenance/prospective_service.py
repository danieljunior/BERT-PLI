from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType

from .persistence_service import PersistenceService

class ProspectiveService:

    DEFAULT_DATAFLOW_TAG = "test_dataflow"

    # Transformation name constants
    TF_FINETUNE_BERT = "finetune_bert"
    TF_TRAIN_PARSE_COLIEE_DATASET = "parse_train_coliee_dataset"
    TF_TRAIN_POOLOUT = "poolout_train"
    TF_TRAIN_PARSE_POOLOUT = "parse_train_poolout"
    TF_TEST_PARSE_COLIEE_DATASET = "parse_coliee_test_dataset"
    TF_TEST_POOLOUT = "poolout_test"
    TF_TEST_PARSE_POOLOUT = "parse_test_poolout"
    TF_TRAIN_CLASSIFIER = "train_classifier"
    TF_TEST_CLASSIFIER = "evaluate"
    TF_PARSE_RESULTS = "parse_results"
    TF_CALCULATE_METRICS = "calculate_metrics"

    # Set name constants
    DT_FINETUNE_CONFIG = "entailment_config"
    DT_FINETUNED_BERT_MODEL = "finetuned_bert_model"
    DT_TRAIN_COLIEE_DATASET = "train_coliee_dataset"
    DT_TRAIN_COLIEE_PARSED_DATASET = "parsed_train_coliee_dataset"
    DT_TRAIN_POOLOUT_CONFIG = "train_poolout_config"
    DT_TRAIN_POOLOUT_DATA = "train_poolout_data"
    DT_TRAIN_PARSED_POOLOUT_DATA = "parsed_train_poolout_data"
    DT_TEST_COLIEE_DATASET = "test_coliee_dataset"
    DT_TEST_COLIEE_PARSED_DATASET = "parsed_test_coliee_dataset"
    DT_TEST_POOLOUT_CONFIG = "test_poolout_config"
    DT_TEST_POOLOUT_DATA = "test_poolout_data"
    DT_TEST_PARSED_POOLOUT_DATA = "parsed_test_poolout_data"
    DT_CLASSIFIER_CONFIG = "classifier_config"
    DT_CLASSIFIER_MODEL = "classifier_model"
    DT_TEST_CONFIG = "evaluate_config"
    DT_TEST_RESULTS = "results"
    DT_PARSED_TEST_RESULTS = "parsed_results"
    DT_TRUE_LABELS = "true_labels"
    DT_METRICS = "metrics"

    def __init__(self, dataflow_tag: str = DEFAULT_DATAFLOW_TAG, persistence_service: PersistenceService = None):
        self.dataflow_tag = dataflow_tag
        self.dataflow = Dataflow(self.dataflow_tag)
        self.persistence_service = persistence_service

    def build_dataflow(self):
        tf_finetune_bert = Transformation(self.TF_FINETUNE_BERT)
        dt_finetuned_config = Set(self.DT_FINETUNE_CONFIG, SetType.INPUT, [
            Attribute("config", AttributeType.TEXT),
            Attribute("checkpoint", AttributeType.FILE)])
        dt_finetuned_bert = Set(self.DT_FINETUNED_BERT_MODEL, SetType.OUTPUT, [
            Attribute("epoch", AttributeType.TEXT),
            Attribute("checkpoint", AttributeType.FILE)])
        tf_finetune_bert.set_sets([dt_finetuned_config, dt_finetuned_bert])
        self.dataflow.add_transformation(tf_finetune_bert)

        tf_parse_coliee = Transformation(self.TF_TRAIN_PARSE_COLIEE_DATASET)
        dt_coliee_dataset = Set(self.DT_TRAIN_COLIEE_DATASET, SetType.INPUT, [
            Attribute("files_path", AttributeType.FILE),
            Attribute("labels_file", AttributeType.FILE),
            Attribute("split_type", AttributeType.TEXT)])
        dt_coliee_parsed_dataset = Set(self.DT_TRAIN_COLIEE_PARSED_DATASET, SetType.OUTPUT, [
            Attribute("vanilla_file", AttributeType.FILE),
            Attribute("summarized_file", AttributeType.FILE),
            Attribute("split_type", AttributeType.TEXT)])
        tf_parse_coliee.set_sets([dt_coliee_dataset, dt_coliee_parsed_dataset])
        self.dataflow.add_transformation(tf_parse_coliee)

        tf_poolout = Transformation(self.TF_TRAIN_POOLOUT)
        dt_coliee_parsed_dataset.set_type(SetType.INPUT)
        dt_coliee_parsed_dataset.dependency = tf_parse_coliee._tag
        dt_finetuned_bert.set_type(SetType.INPUT)
        dt_finetuned_bert.dependency = tf_finetune_bert._tag
        dt_poolout_config = Set(self.DT_TRAIN_POOLOUT_CONFIG, SetType.INPUT, [
            Attribute("config", AttributeType.TEXT),
            Attribute("checkpoint", AttributeType.FILE),
            ])
        dt_poolout_data = Set(self.DT_TRAIN_POOLOUT_DATA, SetType.OUTPUT, [
            Attribute("epoch", AttributeType.TEXT),
            Attribute("poolout_filepath", AttributeType.FILE),
            Attribute("selected_sentences_filepath", AttributeType.FILE)])
        tf_poolout.set_sets([dt_finetuned_bert, dt_coliee_parsed_dataset, dt_poolout_config,
                             dt_poolout_data])
        self.dataflow.add_transformation(tf_poolout)

        tf_parse_poolout = Transformation(self.TF_TRAIN_PARSE_POOLOUT)
        dt_poolout_data.set_type(SetType.INPUT)
        dt_poolout_data.dependency = tf_poolout._tag
        dt_parsed_poolout_data = Set(self.DT_TRAIN_PARSED_POOLOUT_DATA, SetType.OUTPUT, [
            Attribute("filepath", AttributeType.FILE)])
        tf_parse_poolout.set_sets([dt_poolout_data, dt_parsed_poolout_data])
        self.dataflow.add_transformation(tf_parse_poolout)

        tf_train_classifier = Transformation(self.TF_TRAIN_CLASSIFIER)
        dt_parsed_poolout_data.set_type(SetType.INPUT)
        dt_parsed_poolout_data.dependency = tf_parse_poolout._tag
        dt_classifier_config = Set(self.DT_CLASSIFIER_CONFIG, SetType.INPUT, [
            Attribute("config", AttributeType.TEXT)])
        dt_classifier_model = Set(self.DT_CLASSIFIER_MODEL, SetType.OUTPUT, [
            Attribute("epoch", AttributeType.TEXT),
            Attribute("checkpoint", AttributeType.FILE),
            Attribute("validation_metrics_filepath", AttributeType.TEXT),])
        tf_train_classifier.set_sets([dt_parsed_poolout_data, dt_classifier_config, dt_classifier_model])
        self.dataflow.add_transformation(tf_train_classifier)

        tf_test_parse_coliee = Transformation(self.TF_TEST_PARSE_COLIEE_DATASET)
        dt_test_coliee_dataset = Set(self.DT_TEST_COLIEE_DATASET, SetType.INPUT, [
            Attribute("files_path", AttributeType.FILE),
            Attribute("labels_file", AttributeType.FILE),
            Attribute("split_type", AttributeType.TEXT)])
        dt_test_coliee_parsed_dataset = Set(self.DT_TEST_COLIEE_PARSED_DATASET, SetType.OUTPUT, [
            Attribute("vanilla_file", AttributeType.FILE),
            Attribute("summarized_file", AttributeType.FILE),
            Attribute("split_type", AttributeType.TEXT)])
        tf_test_parse_coliee.set_sets([dt_test_coliee_dataset, dt_test_coliee_parsed_dataset])
        self.dataflow.add_transformation(tf_test_parse_coliee)

        tf_test_poolout = Transformation(self.TF_TEST_POOLOUT)
        dt_test_coliee_parsed_dataset.set_type(SetType.INPUT)
        dt_test_coliee_parsed_dataset.dependency = tf_test_parse_coliee._tag
        dt_finetuned_bert.set_type(SetType.INPUT)
        dt_finetuned_bert.dependency = tf_finetune_bert._tag
        dt_test_poolout_config = Set(self.DT_TEST_POOLOUT_CONFIG, SetType.INPUT, [
            Attribute("config", AttributeType.TEXT),
            Attribute("checkpoint", AttributeType.FILE),
            ])
        dt_test_poolout_data = Set(self.DT_TEST_POOLOUT_DATA, SetType.OUTPUT, [
            Attribute("epoch", AttributeType.TEXT),
            Attribute("poolout_filepath", AttributeType.FILE),
            Attribute("selected_sentences_filepath", AttributeType.FILE)])
        tf_test_poolout.set_sets([dt_finetuned_bert, dt_test_coliee_parsed_dataset, dt_test_poolout_config,
                             dt_test_poolout_data])
        self.dataflow.add_transformation(tf_test_poolout)

        tf_test_parse_poolout = Transformation(self.TF_TEST_PARSE_POOLOUT)
        dt_test_poolout_data.set_type(SetType.INPUT)
        dt_test_poolout_data.dependency = tf_test_poolout._tag
        dt_test_parsed_poolout_data = Set(self.DT_TEST_PARSED_POOLOUT_DATA, SetType.OUTPUT, [
            Attribute("filepath", AttributeType.FILE)])
        tf_test_parse_poolout.set_sets([dt_test_poolout_data, dt_test_parsed_poolout_data])
        self.dataflow.add_transformation(tf_test_parse_poolout)

        tf_test = Transformation(self.TF_TEST_CLASSIFIER)
        dt_test_parsed_poolout_data.set_type(SetType.INPUT)
        dt_test_parsed_poolout_data.dependency = tf_test_parse_poolout._tag
        dt_classifier_model.set_type(SetType.INPUT)
        dt_classifier_model.dependency = tf_train_classifier._tag
        dt_test_config = Set(self.DT_TEST_CONFIG, SetType.INPUT, [
            Attribute("config", AttributeType.TEXT),
            Attribute("checkpoint", AttributeType.FILE)])
        dt_test_results = Set(self.DT_TEST_RESULTS, SetType.OUTPUT, [
            Attribute("filepath", AttributeType.FILE)])
        tf_test.set_sets([dt_test_parsed_poolout_data, dt_classifier_model, dt_test_config, dt_test_results])
        self.dataflow.add_transformation(tf_test)

        tf_parse_results = Transformation(self.TF_PARSE_RESULTS)
        dt_test_results.set_type(SetType.INPUT)
        dt_test_results.dependency = tf_test._tag
        dt_parsed_test_results = Set(self.DT_PARSED_TEST_RESULTS, SetType.OUTPUT, [
            Attribute("filepath", AttributeType.FILE)])
        tf_parse_results.set_sets([dt_test_results, dt_parsed_test_results])
        self.dataflow.add_transformation(tf_parse_results)

        tf_calculate_metrics = Transformation(self.TF_CALCULATE_METRICS)
        dt_parsed_test_results.set_type(SetType.INPUT)
        dt_parsed_test_results.dependency = tf_parse_results._tag
        dt_true_labels = Set(self.DT_TRUE_LABELS, SetType.INPUT, [
            Attribute("filepath", AttributeType.FILE)])
        dt_metrics = Set(self.DT_METRICS, SetType.OUTPUT, [
            Attribute("filepath", AttributeType.FILE)])
        tf_calculate_metrics.set_sets([dt_parsed_test_results, dt_true_labels, dt_metrics])
        self.dataflow.add_transformation(tf_calculate_metrics)

        self.dataflow.save()
        self.persistence_service.save_dataflow(self.dataflow_tag, self.dataflow)
        return self.dataflow

if __name__ == "__main__":
    service = ProspectiveService()
    service.build_dataflow()