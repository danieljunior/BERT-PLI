#!/usr/bin/env python3
"""
McNemar Test Script for Pairwise Model Comparison.

This script:
1. Loads ground truth labels from COLIEE format JSON (e.g. data/COLIEE/task1_test_labels_2024.json)
   where each entry is {"query_case.txt": ["candidate_case1.txt", ...]}.
2. Loads prediction results from *_parsed_results.json files with the matching COLIEE format:
   {"query_case.txt": ["retrieved_case1.txt", ...]}.
3. Converts retrieved cases into binary pairs {query_id}_{candidate_id} = 1 (positive/relevant).
4. Evaluates predictions against ground truth (unlisted pairs -> 0 / negative).
5. Constructs 2x2 contingency tables and calculates the McNemar test for all model pairs in each group.
6. Displays the formatted summary table and exports results to a CSV file.

Usage:
    python3 mcnemar_test.py
    python3 mcnemar_test.py --results_dir output/results --groups paragraph summarized vanilla
    python3 mcnemar_test.py --model_a output/results/vanilla/v1_gru_parsed_results.json --model_b output/results/vanilla/v1_lstm_parsed_results.json
"""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


DEFAULT_GROUPS = ["paragraph", "summarized", "vanilla"]


def clean_doc_id(doc_name: str) -> str:
    """Removes trailing .txt or leading/trailing whitespace."""
    doc_name = str(doc_name).strip()
    if doc_name.endswith(".txt"):
        doc_name = doc_name[:-4]
    return doc_name


def format_pair_id(query_id: str, candidate_id: str) -> str:
    """Standardizes pair representation: {query_id}_{candidate_id}."""
    return f"{clean_doc_id(query_id)}_{clean_doc_id(candidate_id)}"


def load_coliee_format(json_path: str) -> Set[str]:
    """
    Loads a JSON file in COLIEE format:
    {
        "085679.txt": ["018047.txt"],
        "086079.txt": ["058944.txt", "078292.txt"]
    }
    or raw prediction format list of [pair_id, scores].

    Returns a set of positive pair strings: 'query_cand' (with .txt removed).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positives = set()
    if isinstance(data, dict):
        for query, candidates in data.items():
            if isinstance(candidates, list):
                for cand in candidates:
                    positives.add(format_pair_id(query, cand))
            elif isinstance(candidates, (int, float)) and candidates > 0:
                positives.add(clean_doc_id(query))
    elif isinstance(data, list):
        for item in data:
            if len(item) == 2:
                pair_id, scores = item
                if isinstance(scores, (list, tuple)):
                    # argmax score: score[1] > score[0] means positive (class 1)
                    if int(np.argmax(scores)) == 1:
                        positives.add(clean_doc_id(pair_id))
                elif scores == 1:
                    positives.add(clean_doc_id(pair_id))

    return positives


def build_aligned_vectors(
    preds_a_positives: Set[str],
    preds_b_positives: Set[str],
    gt_positives: Set[str],
    evaluation_universe: Set[str] = None
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs aligned boolean indicator arrays y_true, y_pred_a, y_pred_b.
    If evaluation_universe is not provided, defaults to all pairs present
    in gt_positives | preds_a_positives | preds_b_positives.
    """
    if evaluation_universe is not None:
        universe = sorted(evaluation_universe)
    else:
        universe = sorted(gt_positives | preds_a_positives | preds_b_positives)

    if not universe:
        raise ValueError("Evaluation universe is empty.")

    y_true = np.array([1 if p in gt_positives else 0 for p in universe], dtype=int)
    y_pred_a = np.array([1 if p in preds_a_positives else 0 for p in universe], dtype=int)
    y_pred_b = np.array([1 if p in preds_b_positives else 0 for p in universe], dtype=int)

    return universe, y_true, y_pred_a, y_pred_b


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates accuracy, precision, recall, and F1."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_contingency_table(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> np.ndarray:
    """
    Constructs the 2x2 contingency table for McNemar's test:

                     Model B Correct    Model B Incorrect
    Model A Correct        b11 (n00)          b10 (n01)
    Model A Incorrect      b01 (n10)          b00 (n11)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b11 = int(np.sum(correct_a & correct_b))
    b10 = int(np.sum(correct_a & ~correct_b))
    b01 = int(np.sum(~correct_a & correct_b))
    b00 = int(np.sum(~correct_a & ~correct_b))

    return np.array([[b11, b10], [b01, b00]], dtype=int)


def run_mcnemar_test(
    table: np.ndarray,
    correction: bool = True
) -> Tuple[float, float, bool]:
    """
    Executes McNemar's test using statsmodels.
    If discordant pairs (b10 + b01) < 25, exact binomial test is used.
    Otherwise, chi-square approximation (with continuity correction by default) is used.

    Returns:
        (statistic, p_value, exact_used)
    """
    b10 = int(table[0, 1])
    b01 = int(table[1, 0])
    discordant = b10 + b01

    if discordant == 0:
        return 0.0, 1.0, False

    exact_used = discordant < 25
    res = mcnemar(table, exact=exact_used, correction=correction)
    stat = float(res.statistic) if res.statistic is not None else 0.0
    pval = float(res.pvalue) if res.pvalue is not None else 1.0

    return stat, pval, exact_used


def scan_results_directory(
    results_dir: str,
    groups: List[str],
    pattern: str = "*_parsed_results.json"
) -> Dict[str, Dict[str, str]]:
    """
    Scans the results directory for parsed prediction files in each group.

    Returns:
        Dict[group_name, Dict[model_name, file_path]]
    """
    found_groups = {}
    base_path = Path(results_dir)

    for group in groups:
        group_dir = base_path / group
        if not group_dir.is_dir():
            continue

        models = {}
        for file_path in sorted(group_dir.glob(pattern)):
            # e.g., v1_gru_parsed_results.json -> v1_gru
            model_name = file_path.stem.replace("_parsed_results", "").replace("_results", "")
            models[model_name] = str(file_path)

        if models:
            found_groups[group] = models

    return found_groups


def compare_pair(
    model_a_name: str,
    model_a_path: str,
    model_b_name: str,
    model_b_path: str,
    gt_positives: Set[str],
    group_name: str = "custom",
    evaluation_universe: Set[str] = None,
    alpha: float = 0.05,
    correction: bool = True
) -> Dict[str, Any]:
    """Performs pairwise comparison between two models."""
    preds_a = load_coliee_format(model_a_path)
    preds_b = load_coliee_format(model_b_path)

    universe, y_true, y_pred_a, y_pred_b = build_aligned_vectors(
        preds_a, preds_b, gt_positives, evaluation_universe=evaluation_universe
    )

    metrics_a = calculate_metrics(y_true, y_pred_a)
    metrics_b = calculate_metrics(y_true, y_pred_b)

    table = compute_contingency_table(y_true, y_pred_a, y_pred_b)
    stat, pval, exact = run_mcnemar_test(table, correction=correction)

    b11, b10 = int(table[0, 0]), int(table[0, 1])
    b01, b00 = int(table[1, 0]), int(table[1, 1])

    # Model advantage assessment
    if pval < alpha:
        if metrics_a["f1"] > metrics_b["f1"]:
            winner = f"{model_a_name} (better F1)"
        elif metrics_b["f1"] > metrics_a["f1"]:
            winner = f"{model_b_name} (better F1)"
        elif metrics_a["accuracy"] > metrics_b["accuracy"]:
            winner = f"{model_a_name} (better acc)"
        elif metrics_b["accuracy"] > metrics_a["accuracy"]:
            winner = f"{model_b_name} (better acc)"
        else:
            winner = "Significant difference (equal metrics)"
    else:
        winner = "No significant difference"

    return {
        "group": group_name,
        "model_A": model_a_name,
        "model_B": model_b_name,
        "total_pairs": len(universe),
        "acc_A": metrics_a["accuracy"],
        "acc_B": metrics_b["accuracy"],
        "p_A": metrics_a["precision"],
        "r_A": metrics_a["recall"],
        "f1_A": metrics_a["f1"],
        "p_B": metrics_b["precision"],
        "r_B": metrics_b["recall"],
        "f1_B": metrics_b["f1"],
        "b11": b11,
        "b10": b10,
        "b01": b01,
        "b00": b00,
        "statistic": stat,
        "p_value": pval,
        "exact": exact,
        "significant": bool(pval < alpha),
        "conclusion": winner,
    }


def print_results_table(results: List[Dict[str, Any]], alpha: float = 0.05) -> None:
    """Formats and prints comparison results table."""
    headers = [
        "Group",
        "Model A",
        "F1 A",
        "Model B",
        "F1 B",
        "b11 (✓✓)",
        "b10 (✓✗)",
        "b01 (✗✓)",
        "b00 (✗✗)",
        "Statistic",
        "p-value",
        f"Sig (p<{alpha})"
    ]

    rows = []
    for r in results:
        sig_str = "YES (*)" if r["significant"] else "NO"
        p_val_str = f"{r['p_value']:.4e}" if r['p_value'] < 0.0001 else f"{r['p_value']:.4f}"
        rows.append([
            r["group"],
            r["model_A"],
            f"{r['f1_A']:.4f}",
            r["model_B"],
            f"{r['f1_B']:.4f}",
            r["b11"],
            r["b10"],
            r["b01"],
            r["b00"],
            f"{r['statistic']:.4f}",
            p_val_str,
            sig_str
        ])

    if HAS_TABULATE:
        print("\n" + tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        header_line = " | ".join(headers)
        print("\n" + "=" * len(header_line))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(c) for c in row))
        print("=" * len(header_line))


def save_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """Saves comparison results to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "group",
        "model_A",
        "model_B",
        "total_pairs",
        "acc_A",
        "acc_B",
        "p_A",
        "r_A",
        "f1_A",
        "p_B",
        "r_B",
        "f1_B",
        "b11",
        "b10",
        "b01",
        "b00",
        "statistic",
        "p_value",
        "exact",
        "significant",
        "conclusion",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Results successfully exported to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="McNemar Test for pairwise comparison of legal case retrieval models using parsed results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="output/results",
        help="Base results directory containing subfolders (default: output/results)"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="data/COLIEE/task1_test_labels_2024.json",
        help="Path to ground truth labels JSON (default: data/COLIEE/task1_test_labels_2024.json)"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        help=f"List of subdirectories to analyze (default: {' '.join(DEFAULT_GROUPS)})"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_parsed_results.json",
        help="Glob pattern for prediction result files (default: *_parsed_results.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/results/mcnemar_results.csv",
        help="Output CSV file path (default: output/results/mcnemar_results.csv)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level threshold (default: 0.05)"
    )
    parser.add_argument(
        "--no_correction",
        action="store_true",
        help="Disable continuity correction in McNemar chi-square test"
    )
    parser.add_argument(
        "--model_a",
        type=str,
        default=None,
        help="Optional path to a single Model A parsed result JSON"
    )
    parser.add_argument(
        "--model_b",
        type=str,
        default=None,
        help="Optional path to a single Model B parsed result JSON"
    )

    args = parser.parse_args()

    # Verify ground truth existence
    if not os.path.isfile(args.ground_truth):
        print(f"Error: Ground truth file not found at: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading ground truth from: {args.ground_truth}")
    gt_positives = load_coliee_format(args.ground_truth)
    print(f"    -> Found {len(gt_positives)} positive pairs in ground truth.\n")

    results: List[Dict[str, Any]] = []
    correction = not args.no_correction

    # Mode 1: Direct comparison between two specific files
    if args.model_a and args.model_b:
        name_a = Path(args.model_a).stem.replace("_parsed_results", "").replace("_results", "")
        name_b = Path(args.model_b).stem.replace("_parsed_results", "").replace("_results", "")
        print(f"[*] Comparing individual models: {name_a} vs {name_b}")
        res = compare_pair(
            model_a_name=name_a,
            model_a_path=args.model_a,
            model_b_name=name_b,
            model_b_path=args.model_b,
            gt_positives=gt_positives,
            group_name="manual",
            alpha=args.alpha,
            correction=correction
        )
        results.append(res)
    else:
        # Mode 2: Batch pairwise comparisons across configured groups
        print(f"[*] Scanning results directory: {args.results_dir} (pattern: {args.pattern})")
        scanned_groups = scan_results_directory(args.results_dir, args.groups, pattern=args.pattern)

        if not scanned_groups:
            print(f"Warning: No valid result files found in {args.results_dir} matching {args.pattern}", file=sys.stderr)
            sys.exit(1)

        for group_name, models in scanned_groups.items():
            print(f"[*] Processing group '{group_name}' ({len(models)} models found: {', '.join(models.keys())})")
            
            # Build unified universe of candidate pairs across this group
            group_universe = set(gt_positives)
            loaded_preds = {}
            for m_name, m_path in models.items():
                m_positives = load_coliee_format(m_path)
                loaded_preds[m_name] = m_positives
                group_universe.update(m_positives)

            model_items = list(models.items())
            for (name_a, path_a), (name_b, path_b) in itertools.combinations(model_items, 2):
                preds_a = loaded_preds[name_a]
                preds_b = loaded_preds[name_b]

                universe, y_true, y_pred_a, y_pred_b = build_aligned_vectors(
                    preds_a, preds_b, gt_positives, evaluation_universe=group_universe
                )

                metrics_a = calculate_metrics(y_true, y_pred_a)
                metrics_b = calculate_metrics(y_true, y_pred_b)

                table = compute_contingency_table(y_true, y_pred_a, y_pred_b)
                stat, pval, exact = run_mcnemar_test(table, correction=correction)

                b11, b10 = int(table[0, 0]), int(table[0, 1])
                b01, b00 = int(table[1, 0]), int(table[1, 1])

                if pval < args.alpha:
                    if metrics_a["f1"] > metrics_b["f1"]:
                        winner = f"{name_a} (better F1)"
                    elif metrics_b["f1"] > metrics_a["f1"]:
                        winner = f"{name_b} (better F1)"
                    elif metrics_a["accuracy"] > metrics_b["accuracy"]:
                        winner = f"{name_a} (better acc)"
                    elif metrics_b["accuracy"] > metrics_a["accuracy"]:
                        winner = f"{name_b} (better acc)"
                    else:
                        winner = "Significant difference (equal metrics)"
                else:
                    winner = "No significant difference"

                results.append({
                    "group": group_name,
                    "model_A": name_a,
                    "model_B": name_b,
                    "total_pairs": len(universe),
                    "acc_A": metrics_a["accuracy"],
                    "acc_B": metrics_b["accuracy"],
                    "p_A": metrics_a["precision"],
                    "r_A": metrics_a["recall"],
                    "f1_A": metrics_a["f1"],
                    "p_B": metrics_b["precision"],
                    "r_B": metrics_b["recall"],
                    "f1_B": metrics_b["f1"],
                    "b11": b11,
                    "b10": b10,
                    "b01": b01,
                    "b00": b00,
                    "statistic": stat,
                    "p_value": pval,
                    "exact": exact,
                    "significant": bool(pval < args.alpha),
                    "conclusion": winner,
                })

    if results:
        print_results_table(results, alpha=args.alpha)
        save_to_csv(results, args.output)
    else:
        print("No comparisons performed.")


if __name__ == "__main__":
    main()
