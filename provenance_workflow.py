#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from pathlib import Path

from config_parser import create_config
from provenance.retrospective_service import RetrospectiveService
from provenance.prospective_service import ProspectiveService

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

import json

def load_env_file(env_file_path):
    """Load environment variables from .run_env file."""
    if not os.path.isfile(env_file_path):
        raise FileNotFoundError(f".env file not found: {env_file_path}")
    
    env_vars = {}
    with open(env_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def validate_env_vars(env_vars):
    """Validate required environment variables."""
    required = [
        'FINETUNE_CONFIG', 'POOLOUT_CONFIG', 'POOLOUT_GPU', 'POOLOUT_TEST_CONFIG',
        'BERT_CHECKPOINT', 'POOLOUT_RESULT', 'POOLOUT_TEST_RESULT', 'TRAIN_LABELS',
        'TRAIN_DATA', 'TEST_DATA', 'LSTM_CONFIG', 'LSTM_GPU', 'LSTM_CHECKPOINT',
        'LSTM_RESULTS', 'GRU_CONFIG', 'GRU_GPU', 'GRU_CHECKPOINT', 'GRU_RESULTS',
        'PARSED_GRU_RESULTS', 'TEST_LABELS', 'GRU_METRICS', 'PARSED_LSTM_RESULTS',
        'LSTM_METRICS'
    ]
    
    missing = [var for var in required if var not in env_vars]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

def read_config_file(file_path):
    """Read config file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def emit_parse_train_coliee(provenance, train_labels_path, mode, output_data):
    """Emit provenance for parsing training COLIEE dataset."""
    task = ProspectiveService.TF_TRAIN_PARSE_COLIEE_DATASET
    dataset_key = ProspectiveService.DT_TRAIN_COLIEE_DATASET
    result_key = ProspectiveService.DT_TRAIN_COLIEE_PARSED_DATASET
    
    input_data = {dataset_key: [[train_labels_path, train_labels_path, mode]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_parse_test_coliee(provenance, test_labels_path, mode, output_data):
    """Emit provenance for parsing testing COLIEE dataset."""
    task = ProspectiveService.TF_TEST_PARSE_COLIEE_DATASET
    dataset_key = ProspectiveService.DT_TEST_COLIEE_DATASET
    result_key = ProspectiveService.DT_TEST_COLIEE_PARSED_DATASET
    
    input_data = {dataset_key: [[test_labels_path, test_labels_path, mode]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_finetune_bert(provenance, finetune_config_path, bert_path, output_data):
    """Emit provenance for BERT finetuning."""
    config_content = read_config_file(finetune_config_path)
    
    task = ProspectiveService.TF_FINETUNE_BERT
    input_key = ProspectiveService.DT_FINETUNE_CONFIG
    output_key = ProspectiveService.DT_FINETUNED_BERT_MODEL
    
    input_data = {input_key: [[config_content, bert_path]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[output_key] = output_data

def emit_poolout_train(provenance, poolout_config_path, bert_checkpoint, output_data):
    """Emit provenance for training poolout."""
    config_content = read_config_file(poolout_config_path)
    
    task = ProspectiveService.TF_TRAIN_POOLOUT
    config_key = ProspectiveService.DT_TRAIN_POOLOUT_CONFIG
    result_key = ProspectiveService.DT_TRAIN_POOLOUT_DATA
    
    input_data = {config_key: [[config_content, bert_checkpoint]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_poolout_to_train_train(provenance, output_data):
    """Emit provenance for poolout_to_train (training)."""
    task = ProspectiveService.TF_TRAIN_PARSE_POOLOUT
    result_key = ProspectiveService.DT_TRAIN_PARSED_POOLOUT_DATA
    
    input_data = {}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_poolout_test(provenance, poolout_test_config_path, bert_checkpoint, output_data):
    """Emit provenance for testing poolout."""
    config_content = read_config_file(poolout_test_config_path)
    
    task = ProspectiveService.TF_TEST_POOLOUT
    config_key = ProspectiveService.DT_TEST_POOLOUT_CONFIG
    result_key = ProspectiveService.DT_TEST_POOLOUT_DATA
    
    input_data = {config_key: [[config_content, bert_checkpoint]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_poolout_to_train_test(provenance, output_data):
    """Emit provenance for poolout_to_train (testing)."""
    task = ProspectiveService.TF_TEST_PARSE_POOLOUT
    result_key = ProspectiveService.DT_TEST_PARSED_POOLOUT_DATA
    
    input_data = {}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_train_classifier(provenance, classifier_config_path, output_data):
    """Emit provenance for training classifier (LSTM/GRU)."""
    config_content = read_config_file(classifier_config_path)
    
    task = ProspectiveService.TF_TRAIN_CLASSIFIER
    config_key = ProspectiveService.DT_CLASSIFIER_CONFIG
    result_key = ProspectiveService.DT_CLASSIFIER_MODEL
    
    input_data = {config_key: [[config_content]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_test_classifier(provenance, classifier_config_path, classifier_checkpoint, output_data):
    """Emit provenance for testing classifier (LSTM/GRU)."""
    config_content = read_config_file(classifier_config_path)
    
    task = ProspectiveService.TF_TEST_CLASSIFIER
    config_key = ProspectiveService.DT_TEST_CONFIG
    result_key = ProspectiveService.DT_TEST_RESULTS
    
    input_data = {config_key: [[config_content, classifier_checkpoint]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_parse_results(provenance, output_data):
    """Emit provenance for parsing results."""
    task = ProspectiveService.TF_PARSE_RESULTS
    result_key = ProspectiveService.DT_PARSED_TEST_RESULTS
    
    input_data = {}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def emit_calculate_metrics(provenance, test_labels_path, output_data):
    """Emit provenance for calculating metrics."""
    task = ProspectiveService.TF_CALCULATE_METRICS
    labels_key = ProspectiveService.DT_TRUE_LABELS
    result_key = ProspectiveService.DT_METRICS
    
    input_data = {labels_key: [[test_labels_path]]}
    
    with provenance.get_retrospective_data(task, input_data) as result:
        result[result_key] = output_data

def load_classifier_output(metrics_path, model_type):
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return []
        
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output = []
        for result in data.get("results", []):
            epoch = str(result["epoch"])
            checkpoint = result["checkpoint"]
            metrics_str = json.dumps(result.get("metrics", {}))
            output.append([epoch, "/app/"+checkpoint, metrics_str])
            
        return output
    except Exception as e:
        logger.error(f"Error parsing metrics file {metrics_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description='Emit provenance directives for BERT-PLI workflow in run_workflow.sh order'
    )
    parser.add_argument('dataflow_tag', help='Unique dataflow tag for provenance tracking')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(script_dir, '.run_env')
    
    env_vars = load_env_file(env_file)
    validate_env_vars(env_vars)
    
    os.environ['DATAFLOW_TAG'] = args.dataflow_tag
    
    logger.info(f"Using DATAFLOW_TAG: {args.dataflow_tag}")
    logger.info("Initializing provenance service...")
    
    provenance = RetrospectiveService(args.dataflow_tag)
    
    config_finetune = create_config(env_vars['FINETUNE_CONFIG'])
    bert_path = config_finetune.get("model", "bert_path")
    
    config_poolout = create_config(env_vars['POOLOUT_CONFIG'])
    train_sentences_file = config_poolout.get("data", "train_data_path") + "/" + config_poolout.get("data", "train_file_list")
    
    config_poolout_test = create_config(env_vars['POOLOUT_TEST_CONFIG'])
    test_sentences_file = config_poolout_test.get("data", "test_data_path") + "/" + config_poolout_test.get("data", "test_file_list")
    
    train_vanilla_output = env_vars['TRAIN_LABELS'] + '_vanilla'
    train_summarized_output = env_vars['TRAIN_LABELS'] + '_summarized'
    test_vanilla_output = env_vars['TEST_LABELS'] + '_vanilla'
    test_summarized_output = env_vars['TEST_LABELS'] + '_summarized'
    
    parse_train_coliee_output = [[train_vanilla_output, train_summarized_output, 'train']]
    parse_test_coliee_output = [[test_vanilla_output, test_summarized_output, 'test']]
    finetune_bert_output = [['1', env_vars['BERT_CHECKPOINT']], ['2', env_vars['BERT_CHECKPOINT']]]
    poolout_train_output = [['1', env_vars['POOLOUT_RESULT'], train_sentences_file]]
    poolout_to_train_train_output = [[env_vars['TRAIN_DATA']]]
    poolout_test_output = [['1', env_vars['POOLOUT_TEST_RESULT'], test_sentences_file]]
    poolout_to_train_test_output = [[env_vars['TEST_DATA']]]
    train_lstm_output = load_classifier_output('output/results/summarized/attenlstm_valid_metrics.json', 'lstm')
    if not train_lstm_output:
        train_lstm_output = [['1', 'output/checkpoints/lstm/1.pkl', 'output/validation_1_lstm.json'],
                             ['2', 'output/checkpoints/lstm/2.pkl', 'output/validation_2_lstm.json']]
    test_lstm_output = [[env_vars['LSTM_RESULTS']]]
    parse_lstm_results_output = [[env_vars['PARSED_LSTM_RESULTS']]]
    calculate_lstm_metrics_output = [[env_vars['LSTM_METRICS']]]
    train_gru_output = load_classifier_output('output/results/summarized/attengru_valid_metrics.json', 'gru')
    if not train_gru_output:
        train_gru_output = [['1', 'output/checkpoints/gru/1.pkl', 'output/validation_1_gru.json'],
                            ['2', 'output/checkpoints/gru/2.pkl', 'output/validation_2_gru.json']]
    test_gru_output = [[env_vars['GRU_RESULTS']]]
    parse_gru_results_output = [[env_vars['PARSED_GRU_RESULTS']]]
    calculate_gru_metrics_output = [[env_vars['GRU_METRICS']]]
    
    try:
        logger.info("=" * 70)
        logger.info("Step 1: Parse training COLIEE dataset")
        logger.info("=" * 70)
        emit_parse_train_coliee(provenance, env_vars['TRAIN_LABELS'], 'train', parse_train_coliee_output)
        
        logger.info("=" * 70)
        logger.info("Step 2: Parse testing COLIEE dataset")
        logger.info("=" * 70)
        emit_parse_test_coliee(provenance, env_vars['TEST_LABELS'], 'test', parse_test_coliee_output)
        
        logger.info("=" * 70)
        logger.info("Step 3: Finetune BERT")
        logger.info("=" * 70)
        emit_finetune_bert(provenance, env_vars['FINETUNE_CONFIG'], bert_path, finetune_bert_output)
        
        logger.info("=" * 70)
        logger.info("Step 4: Poolout (training)")
        logger.info("=" * 70)
        emit_poolout_train(provenance, env_vars['POOLOUT_CONFIG'], env_vars['BERT_CHECKPOINT'], poolout_train_output)
        
        logger.info("=" * 70)
        logger.info("Step 5: Poolout-to-train (training)")
        logger.info("=" * 70)
        emit_poolout_to_train_train(provenance, poolout_to_train_train_output)
        
        logger.info("=" * 70)
        logger.info("Step 6: Poolout (testing)")
        logger.info("=" * 70)
        emit_poolout_test(provenance, env_vars['POOLOUT_TEST_CONFIG'], env_vars['BERT_CHECKPOINT'], poolout_test_output)
        
        logger.info("=" * 70)
        logger.info("Step 7: Poolout-to-train (testing)")
        logger.info("=" * 70)
        emit_poolout_to_train_test(provenance, poolout_to_train_test_output)
        
        logger.info("=" * 70)
        logger.info("Step 8: Train LSTM classifier")
        logger.info("=" * 70)
        emit_train_classifier(provenance, env_vars['LSTM_CONFIG'], train_lstm_output)
        
        logger.info("=" * 70)
        logger.info("Step 9: Test LSTM classifier")
        logger.info("=" * 70)
        emit_test_classifier(provenance, env_vars['LSTM_CONFIG'], env_vars['LSTM_CHECKPOINT'], test_lstm_output)
        
        logger.info("=" * 70)
        logger.info("Step 10: Parse LSTM results")
        logger.info("=" * 70)
        emit_parse_results(provenance, parse_lstm_results_output)
        
        logger.info("=" * 70)
        logger.info("Step 11: Calculate LSTM metrics")
        logger.info("=" * 70)
        emit_calculate_metrics(provenance, env_vars['TEST_LABELS'], calculate_lstm_metrics_output)
        
        logger.info("=" * 70)
        logger.info("Step 12: Train GRU classifier")
        logger.info("=" * 70)
        emit_train_classifier(provenance, env_vars['GRU_CONFIG'], train_gru_output)
        
        logger.info("=" * 70)
        logger.info("Step 13: Test GRU classifier")
        logger.info("=" * 70)
        emit_test_classifier(provenance, env_vars['GRU_CONFIG'], env_vars['GRU_CHECKPOINT'], test_gru_output)
        
        logger.info("=" * 70)
        logger.info("Step 14: Parse GRU results")
        logger.info("=" * 70)
        emit_parse_results(provenance, parse_gru_results_output)
        
        logger.info("=" * 70)
        logger.info("Step 15: Calculate GRU metrics")
        logger.info("=" * 70)
        emit_calculate_metrics(provenance, env_vars['TEST_LABELS'], calculate_gru_metrics_output)
        
        logger.info("=" * 70)
        logger.info("All provenance directives emitted successfully")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error during provenance emission: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
