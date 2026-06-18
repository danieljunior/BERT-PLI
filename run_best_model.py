#!/usr/bin/env python3
"""
Select best checkpoint from validation metrics and run test -> parse -> evaluate.

Usage example:
  ./run_best_model.py --metrics-file metrics.json -c config.ini -g 0 \
      --results out.json --parsed-results parsed.json --test-labels labels.json --final-metrics final.json
"""
import argparse
import json
import sys
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run best model (test -> parse -> evaluate) based on metrics JSON")
    p.add_argument('--metrics-file', required=True, help='Path to JSON file containing validation metrics')
    p.add_argument('-c', '--config', required=True, help='Config file path')
    p.add_argument('-g', '--gpu', default=0, help='GPU id (default 0)')
    p.add_argument('--results', required=True, help='Path for output test results')
    p.add_argument('--parsed-results', required=True, help='Path for parsed results')
    p.add_argument('--test-labels', required=True, help='Path for test labels')
    p.add_argument('--final-metrics', required=True, help='Path for final evaluated metrics')
    return p.parse_args()


def load_metrics(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    try:
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in metrics file {path}: {e}")
    return data


def find_best_checkpoint(metrics_data):
    if not isinstance(metrics_data, dict):
        raise ValueError('Metrics file root should be a JSON object')
    results = metrics_data.get('results')
    if not results or not isinstance(results, list):
        raise ValueError('No results list found in metrics file')

    best_item = None
    best_f1 = float('-inf')
    for item in results:
        if not isinstance(item, dict):
            continue
        metrics = item.get('metrics') or {}
        # Ensure f1 exists and is numeric
        if 'f1' not in metrics:
            continue
        try:
            f1 = float(metrics['f1'])
        except Exception:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_item = item

    if best_item is None:
        raise ValueError('No valid result with numeric metrics.f1 was found in metrics file')

    checkpoint = best_item.get('checkpoint')
    if not checkpoint:
        raise ValueError('Best item does not contain a checkpoint path')

    return checkpoint, best_f1


def run_command(cmd_list, desc=None):
    if desc:
        print(f"Running: {desc} -> {' '.join(cmd_list)}")
    else:
        print(f"Running: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)


def main():
    args = parse_args()

    try:
        metrics = load_metrics(args.metrics_file)
    except Exception as e:
        print(f"Error loading metrics file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        checkpoint, best_f1 = find_best_checkpoint(metrics)
    except Exception as e:
        print(f"Error selecting best checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Selected checkpoint: {checkpoint} (f1={best_f1})")

    # Build commands (use same Python interpreter)
    py = sys.executable or 'python3'
    cmd_test = [py, 'test.py', '-c', args.config, '-g', str(args.gpu), '--checkpoint', checkpoint, '--result', args.results]
    cmd_parse = [py, 'parse_results.py', 'parse', args.results, args.parsed_results]
    cmd_eval = [py, 'parse_results.py', 'evaluate', args.test_labels, args.parsed_results, args.final_metrics]

    try:
        run_command(cmd_test, desc='test.py')
        run_command(cmd_parse, desc='parse_results.py parse')
        run_command(cmd_eval, desc='parse_results.py evaluate')
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {e}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
