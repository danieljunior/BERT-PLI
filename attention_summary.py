#!/usr/bin/env python3
"""
Attention Summary Script — Quantitative attention weight analysis (AttenRNN: GRU/LSTM).

This script:
1. Loads validation metrics JSON and finds the best checkpoint via run_best_model.find_best_checkpoint.
2. Translates Docker paths (/app/...) to local paths.
3. Loads the test dataset and AttenRNN model (GRU/LSTM).
4. Recomputes softmax attention weights [B, M] over candidate document segments.
5. Calculates quantitative metrics per sample:
   - Entropy
   - Max weight
   - Argmax weight (index of max focused segment)
   - Min weight
   - Top-K concentration sum
   - Gini coefficient of inequality
6. Exports per-sample CSV (pair_stats.csv), aggregate JSON (aggregate_stats.json), and optional histogram PNGs.

Usage:
    python3 attention_summary.py \
        --config config/nlp/divergent/vanilla_gru.config \
        --metrics output/results/vanilla/v1_attengru_valid_metrics.json \
        --output output/attention_stats/vanilla_gru \
        [--ground_truth data/COLIEE/task1_test_labels_2024.json] \
        [--plots] [--top_k 3] [--gpu 0] [--path_prefix /app]
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from config_parser.parser import create_config
from tools.init_tool import init_all
from run_best_model import load_metrics, find_best_checkpoint


# ─── Attention Quantitative Metrics ───────────────────────────────────

def compute_attention_stats(attn: np.ndarray, top_k: int = 3) -> Dict[str, np.ndarray]:
    """
    Computes quantitative metrics per sample from attention softmax array [B, M].

    Args:
        attn: [B, M] array of softmax attention weights summing to 1 across M.
        top_k: Top-K elements to sum for concentration metric.

    Returns:
        Dict mapping metric names to [B] numpy arrays.
    """
    eps = 1e-9
    B, M = attn.shape

    # Shannon Entropy H = - sum(w * log(w))
    entropy = -np.sum(attn * np.log(attn + eps), axis=-1)

    # Max, Argmax, Min
    max_weight = attn.max(axis=-1)
    argmax_w = attn.argmax(axis=-1)
    min_weight = attn.min(axis=-1)

    # Top-K concentration
    top_k_clamped = min(top_k, M)
    sorted_desc = np.sort(attn, axis=-1)[:, ::-1]
    topk_conc = sorted_desc[:, :top_k_clamped].sum(axis=-1)

    # Gini Index: (2 * sum_{i=1}^M i * w_(i)) / (M * sum w_i) - (M + 1) / M
    sorted_asc = np.sort(attn, axis=-1)
    idx = np.arange(1, M + 1)
    total_w = sorted_asc.sum(axis=-1)
    gini = (2.0 * np.sum(idx * sorted_asc, axis=-1)) / (M * total_w + eps) - (M + 1.0) / M
    gini = np.clip(gini, 0.0, 1.0)

    return {
        "entropy": entropy,
        "max_weight": max_weight,
        "argmax_weight": argmax_w,
        "min_weight": min_weight,
        f"top{top_k}_concentration": topk_conc,
        "gini": gini,
    }


# ─── RNN Attention Extraction ──────────────────────────────────────────

def extract_rnn_attention(model, data, config, gpu_list) -> np.ndarray:
    """
    Recalculates softmax attention weights for AttenRNN models.
    Matches the forward mathematical operations in AttenRNN.Attention without modifying the class.
    """
    model.eval()
    with torch.no_grad():
        x = data['input']  # [B, M, I]
        batch_size = x.size(0)

        model.init_hidden(config, batch_size, gpu_list)
        rnn_out, _ = model.rnn(x, model.hidden)            # [B, M, 2H]
        tmp_rnn = rnn_out.permute(0, 2, 1)              # [B, 2H, M]
        feature = model.max_pool(tmp_rnn).squeeze(2)    # [B, 2H]
        feature = model.fc_a(feature).unsqueeze(2)      # [B, 2H, 1]

        ratio = torch.bmm(rnn_out, feature)             # [B, M, 1]
        ratio = ratio.view(ratio.size(0), ratio.size(1)) # [B, M]
        attn_w = F.softmax(ratio, dim=1)                # [B, M]

        return attn_w.cpu().numpy()


# ─── Ground Truth Loader ───────────────────────────────────────────────

def clean_doc_id(doc_name: str) -> str:
    doc_name = str(doc_name).strip()
    if doc_name.endswith(".txt"):
        doc_name = doc_name[:-4]
    return doc_name


def load_gt_positives(gt_path: str) -> Set[str]:
    """Loads positive pair IDs ('query_candidate') from COLIEE ground truth JSON."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positives = set()
    for query, candidates in data.items():
        q_id = clean_doc_id(query)
        for cand in candidates:
            positives.add(f"{q_id}_{clean_doc_id(cand)}")

    return positives


# ─── Exporters ─────────────────────────────────────────────────────────

def save_csv(records: List[Dict[str, Any]], output_dir: str) -> str:
    path = os.path.join(output_dir, "pair_stats.csv")
    if not records:
        return path

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[+] Saved per-pair CSV to: {path}")
    return path


def save_json(records: List[Dict[str, Any]], output_dir: str, top_k: int) -> str:
    path = os.path.join(output_dir, "aggregate_stats.json")
    if not records:
        return path

    topk_key = f"top{top_k}_concentration"
    metric_keys = ["entropy", "max_weight", "min_weight", topk_key, "gini"]

    result = {
        "n_samples": len(records),
    }

    for key in metric_keys:
        vals = np.array([r[key] for r in records], dtype=float)
        result[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Argmax frequency distribution
    argmax_vals = [r["argmax_weight"] for r in records]
    argmax_counts = dict(Counter(argmax_vals))
    # Convert keys to string for clean JSON serialization
    result["argmax_weight"] = {
        "value_counts": {str(k): v for k, v in sorted(argmax_counts.items())}
    }

    # Breakdown by label if ground truth is present
    if "gt_label" in records[0]:
        pos_records = [r for r in records if r["gt_label"] == 1]
        neg_records = [r for r in records if r["gt_label"] == 0]

        result["by_label"] = {
            "positive_count": len(pos_records),
            "negative_count": len(neg_records),
            "positive": {},
            "negative": {},
        }

        for key in metric_keys:
            if pos_records:
                p_vals = np.array([r[key] for r in pos_records], dtype=float)
                result["by_label"]["positive"][key] = {
                    "mean": float(np.mean(p_vals)),
                    "std": float(np.std(p_vals)),
                    "median": float(np.median(p_vals)),
                }
            if neg_records:
                n_vals = np.array([r[key] for r in neg_records], dtype=float)
                result["by_label"]["negative"][key] = {
                    "mean": float(np.mean(n_vals)),
                    "std": float(np.std(n_vals)),
                    "median": float(np.median(n_vals)),
                }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[+] Saved aggregate JSON to: {path}")
    return path


def plot_distributions(records: List[Dict[str, Any]], output_dir: str, top_k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot generation.")
        return

    plots_dir = os.path.join(output_dir, "histograms")
    os.makedirs(plots_dir, exist_ok=True)

    topk_key = f"top{top_k}_concentration"
    metric_keys = ["entropy", "max_weight", "min_weight", topk_key, "gini"]

    for key in metric_keys:
        vals = np.array([r[key] for r in records], dtype=float)
        plt.figure(figsize=(8, 4))
        plt.hist(vals, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(f"Distribution of {key}")
        plt.xlabel(key)
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname = os.path.join(plots_dir, f"{key}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    # Argmax distribution bar plot
    argmax_vals = [r["argmax_weight"] for r in records]
    counts = Counter(argmax_vals)
    sorted_items = sorted(counts.items())

    if sorted_items:
        x_vals, y_vals = zip(*sorted_items)
        plt.figure(figsize=(max(8, len(x_vals) // 2), 4))
        plt.bar([str(x) for x in x_vals], y_vals, color="coral", edgecolor="black", alpha=0.7)
        plt.title("Argmax Weight Distribution (Focused Segment Index)")
        plt.xlabel("Segment Index (argmax)")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fname = os.path.join(plots_dir, "argmax_distribution.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    print(f"[+] Histograms saved in: {plots_dir}")


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative attention summary script for AttenRNN (GRU/LSTM) models."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to model config file (.config)")
    parser.add_argument(
        "--metrics", "-m", required=True,
        help="Path to validation metrics JSON (structured like run_best_model.py output)"
    )
    parser.add_argument("--output", "-o", required=True, help="Directory to save output files")
    parser.add_argument(
        "--ground_truth", default=None,
        help="Path to COLIEE ground truth JSON (optional, adds gt_label column)"
    )
    parser.add_argument("--top_k", type=int, default=3, help="Top-K count for concentration metric (default: 3)")
    parser.add_argument("--gpu", default=None, help="GPU ID list (e.g. 0). Omit to run on CPU.")
    parser.add_argument(
        "--path_prefix", default="/app",
        help="Prefix in Docker paths to be replaced with '.' (default: /app)"
    )
    parser.add_argument("--plots", action="store_true", help="Generate PNG histogram plots")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Parse GPU configuration
    gpu_list = []
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        for item in args.gpu.split(","):
            gpu_list.append(int(item))

    # 2. Parse config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = create_config(args.config)
    model_name = config.get("model", "model_name")
    if model_name != "AttenRNN":
        print(
            f"Error: This script only supports AttenRNN models (GRU/LSTM). Config specifies '{model_name}'.",
            file=sys.stderr
        )
        sys.exit(1)

    rnn_type = config.get("model", "rnn")
    print(f"[*] Architecture: AttenRNN ({rnn_type.upper()})")

    # 3. Load metrics JSON & find best checkpoint dynamically
    print(f"[*] Loading validation metrics from: {args.metrics}")
    metrics_data = load_metrics(args.metrics)
    checkpoint_docker, best_f1 = find_best_checkpoint(metrics_data)

    # Translate Docker path (/app/...) to local path (./...)
    if args.path_prefix and checkpoint_docker.startswith(args.path_prefix):
        checkpoint = "." + checkpoint_docker[len(args.path_prefix):]
    else:
        checkpoint = checkpoint_docker

    if not os.path.exists(checkpoint):
        # Fallback check if relative path exists directly
        if os.path.exists(checkpoint_docker):
            checkpoint = checkpoint_docker
        else:
            print(f"Error: Checkpoint file not found at: {checkpoint} (original: {checkpoint_docker})", file=sys.stderr)
            sys.exit(1)

    print(f"[*] Best checkpoint selected: {checkpoint} (Validation F1 = {best_f1:.4f})")

    # 4. Initialize model & test dataloader
    print("[*] Initializing model and test dataset...")
    parameters = init_all(config, gpu_list, checkpoint, "test")
    model = parameters["model"]
    dataloader = parameters["test_dataset"]
    model.eval()

    # 5. Load Ground Truth if provided
    gt_positives = None
    if args.ground_truth:
        if os.path.exists(args.ground_truth):
            gt_positives = load_gt_positives(args.ground_truth)
            print(f"[*] Loaded {len(gt_positives)} ground truth positive pairs from: {args.ground_truth}")
        else:
            print(f"Warning: Ground truth file not found at {args.ground_truth}, ignoring.", file=sys.stderr)

    # 6. Run inference and extract attention statistics
    records = []
    print(f"[*] Extracting attention stats across {len(dataloader)} batches...")
    for data in tqdm(dataloader):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0 and torch.cuda.is_available():
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        attn_w = extract_rnn_attention(model, data, config, gpu_list)  # [B, M]
        stats = compute_attention_stats(attn_w, top_k=args.top_k)      # Dict of [B] arrays

        for i, guid in enumerate(data['guid']):
            row = {"guid": guid}
            for k, arr in stats.items():
                val = arr[i]
                if k == "argmax_weight":
                    row[k] = int(val)
                else:
                    row[k] = float(round(float(val), 6))

            if gt_positives is not None:
                clean_guid = clean_doc_id(guid)
                row["gt_label"] = 1 if clean_guid in gt_positives else 0

            records.append(row)

    print(f"[*] Completed extraction for {len(records)} test pairs.")

    # 7. Save CSV & JSON outputs
    save_csv(records, args.output)
    save_json(records, args.output, top_k=args.top_k)

    if args.plots:
        plot_distributions(records, args.output, top_k=args.top_k)

    print("\n[+] Attention summary successfully completed!")


if __name__ == "__main__":
    main()
