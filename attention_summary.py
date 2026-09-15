#!/usr/bin/env python3
"""
Quantitative Attention Weight Summarizer for Divergent Model Predictions.

Operates on the same divergent prediction subsets as attention_divergent_predictions.py,
calculating Shannon entropy, max weight, argmax weight, min weight, top-K concentration,
and Gini coefficient across divergent evaluation samples.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from attention_metrics import (
    compute_attention_metrics,
    compute_numeric_summary,
    summarize_frequency_counts,
)
from divergent_subset_loader import (
    calculate_intra_divergent_pairs,
    calculate_inter_divergent_pairs,
    extract_target_guid_set,
    collect_filtered_indices,
)
from run_best_model import load_metrics, find_best_checkpoint


def clean_identifier_stem(identifier_str: str) -> str:
    """Strip .txt suffix from document identifier strings."""
    stripped = str(identifier_str).strip()
    if stripped.endswith(".txt"):
        return stripped[:-4]
    return stripped


def load_ground_truth_label_set(gt_json_path: str) -> set[str]:
    """Load set of positive pair identifiers '{query}_{candidate}' from ground truth."""
    path_obj = Path(gt_json_path)
    if not path_obj.is_file():
        return set()
    with path_obj.open("r", encoding="utf-8") as file_handle:
        gt_data = json.load(file_handle)
    positives_set: set[str] = set()
    for query_doc, candidate_list in gt_data.items():
        q_clean = clean_identifier_stem(query_doc)
        for cand_doc in candidate_list:
            c_clean = clean_identifier_stem(cand_doc)
            positives_set.add(f"{q_clean}_{c_clean}")
    return positives_set


def resolve_validation_metrics_file(
    variant: str, model_name: str, version: str
) -> str:
    """Find valid metrics JSON path for a given variant, model, and version."""
    candidate_paths = [
        f"output/results/{variant}/{version}_atten{model_name}_valid_metrics.json",
        f"output/results/v1/{variant}/{version}_atten{model_name}_valid_metrics.json",
        f"output/results/{variant}/{version}_{model_name}_metrics.json",
    ]
    for path_str in candidate_paths:
        if os.path.isfile(path_str):
            return path_str
    msg = f"No valid metrics JSON found for {variant}/{model_name} in {candidate_paths}"
    raise FileNotFoundError(msg)


def resolve_best_checkpoint_file(metrics_path: str, path_prefix: str) -> str:
    """Retrieve best checkpoint path from validation metrics JSON."""
    metrics_data = load_metrics(metrics_path)
    raw_checkpoint_path, _ = find_best_checkpoint(metrics_data)
    if path_prefix and raw_checkpoint_path.startswith(path_prefix):
        return "." + raw_checkpoint_path[len(path_prefix):]
    return raw_checkpoint_path


def export_pair_records_csv(
    record_rows: list[dict[str, str | int | float]], target_path: str
) -> None:
    """Export per-sample attention metrics table to CSV file."""
    if not record_rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    field_names = list(record_rows[0].keys())
    with open(target_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(record_rows)
    print(f"[+] Saved per-pair CSV to: {target_path}")


def build_label_breakdown(
    record_rows: list[dict[str, str | int | float]], metric_keys: list[str]
) -> dict[str, object]:
    """Group summary metrics by positive and negative ground truth label."""
    pos_items = [r for r in record_rows if r.get("gt_label") == 1]
    neg_items = [r for r in record_rows if r.get("gt_label") == 0]
    breakdown: dict[str, object] = {
        "positive_count": len(pos_items),
        "negative_count": len(neg_items),
        "positive": {},
        "negative": {},
    }
    for metric_name in metric_keys:
        if pos_items:
            pos_vals = np.array([row[metric_name] for row in pos_items], dtype=float)
            breakdown["positive"][metric_name] = compute_numeric_summary(pos_vals)
        if neg_items:
            neg_vals = np.array([row[metric_name] for row in neg_items], dtype=float)
            breakdown["negative"][metric_name] = compute_numeric_summary(neg_vals)
    return breakdown


def export_aggregate_metrics_json(
    record_rows: list[dict[str, str | int | float]], target_path: str, k_value: int
) -> None:
    """Export statistical aggregation of attention metrics to JSON file."""
    if not record_rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    metric_keys = [
        "entropy",
        "max_weight",
        "min_weight",
        f"top{k_value}_concentration",
        "gini",
    ]
    summary_doc: dict[str, object] = {"total_samples": len(record_rows)}
    for metric_name in metric_keys:
        val_array = np.array([row[metric_name] for row in record_rows], dtype=float)
        summary_doc[metric_name] = compute_numeric_summary(val_array)
    argmax_integers = [int(row["argmax_weight"]) for row in record_rows]
    summary_doc["argmax_weight"] = summarize_frequency_counts(argmax_integers)
    if "gt_label" in record_rows[0]:
        summary_doc["by_label"] = build_label_breakdown(record_rows, metric_keys)
    with open(target_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary_doc, file_handle, indent=2, ensure_ascii=False)
    print(f"[+] Saved aggregate JSON to: {target_path}")


def render_metric_plot(
    values: np.ndarray, metric_title: str, destination_path: str
) -> None:
    """Render and save a single metric distribution histogram."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=40, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(f"Distribution of {metric_title}")
    plt.xlabel(metric_title)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(destination_path, dpi=150)
    plt.close()


def render_argmax_plot(
    argmax_counts: dict[str, int], destination_path: str
) -> None:
    """Render and save argmax categorical bar distribution."""
    import matplotlib.pyplot as plt

    sorted_tuples = sorted(argmax_counts.items(), key=lambda t: int(t[0]))
    if not sorted_tuples:
        return
    x_positions, y_counts = zip(*sorted_tuples)
    fig_width = max(8, len(x_positions) // 2)
    plt.figure(figsize=(fig_width, 4))
    plt.bar(x_positions, y_counts, color="coral", edgecolor="black", alpha=0.7)
    plt.title("Argmax Weight Distribution (Focused Segment Index)")
    plt.xlabel("Segment Index")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(destination_path, dpi=150)
    plt.close()


def export_distribution_plots(
    record_rows: list[dict[str, str | int | float]], plots_dir: str, k_value: int
) -> None:
    """Generate distribution histogram images for all attention metrics."""
    os.makedirs(plots_dir, exist_ok=True)
    metric_keys = [
        "entropy",
        "max_weight",
        "min_weight",
        f"top{k_value}_concentration",
        "gini",
    ]
    for metric_name in metric_keys:
        num_vals = np.array([r[metric_name] for r in record_rows], dtype=float)
        img_path = os.path.join(plots_dir, f"{metric_name}.png")
        render_metric_plot(num_vals, metric_name, img_path)
    argmax_list = [int(row["argmax_weight"]) for row in record_rows]
    counts_map = summarize_frequency_counts(argmax_list)
    argmax_img = os.path.join(plots_dir, "argmax_distribution.png")
    render_argmax_plot(counts_map, argmax_img)
    print(f"[+] Histograms saved in: {plots_dir}")


def extract_sample_record(
    guid_str: str,
    metric_arrays: dict[str, np.ndarray],
    batch_index: int,
    gt_positives: set[str] | None,
) -> dict[str, str | int | float]:
    """Construct one sample metrics row dictionary from computed arrays."""
    sample_row: dict[str, str | int | float] = {"guid": guid_str}
    for metric_name, values_array in metric_arrays.items():
        sample_value = values_array[batch_index]
        if metric_name == "argmax_weight":
            sample_row[metric_name] = int(sample_value)
        else:
            sample_row[metric_name] = float(round(float(sample_value), 6))
    if gt_positives is not None:
        clean_guid = clean_identifier_stem(guid_str)
        sample_row["gt_label"] = 1 if clean_guid in gt_positives else 0
    return sample_row


def extract_rnn_softmax_weights(
    model: object, data_batch: dict[str, object], config: object, gpu_list: list[int]
) -> np.ndarray:
    """Recalculate softmax attention weights [B, M] for AttenRNN models."""
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        x_tensor = data_batch["input"]
        batch_size = x_tensor.size(0)
        model.init_hidden(config, batch_size, gpu_list)
        rnn_out, _ = model.rnn(x_tensor, model.hidden)
        tmp_rnn = rnn_out.permute(0, 2, 1)
        feature = model.max_pool(tmp_rnn).squeeze(2)
        feature = model.fc_a(feature).unsqueeze(2)
        ratio = torch.bmm(rnn_out, feature)
        ratio = ratio.view(ratio.size(0), ratio.size(1))
        softmax_weights = F.softmax(ratio, dim=1)
        return softmax_weights.cpu().numpy()


def create_filtered_dataloader(
    dataloader: object, divergent_pairs: list[tuple[str, str]]
) -> object:
    """Filter PyTorch DataLoader to contain only divergent sample pairs."""
    from torch.utils.data import DataLoader, Subset

    target_guid_set = extract_target_guid_set(divergent_pairs)
    dataset = dataloader.dataset
    matching_indices = collect_filtered_indices(dataset, target_guid_set)
    subset_dataset = Subset(dataset, matching_indices)
    return DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers,
        shuffle=False,
    )


def evaluate_divergent_dataloader(
    model: object,
    filtered_loader: object,
    config: object,
    gpu_list: list[int],
    gt_positives: set[str] | None,
    k_value: int,
) -> list[dict[str, str | int | float]]:
    """Iterate through filtered DataLoader and compute attention metrics per sample."""
    import torch
    from torch.autograd import Variable
    from tqdm import tqdm

    collected_rows: list[dict[str, str | int | float]] = []
    for batch_item in tqdm(filtered_loader, desc="Extracting attention"):
        for tensor_key in batch_item.keys():
            if isinstance(batch_item[tensor_key], torch.Tensor):
                if len(gpu_list) > 0 and torch.cuda.is_available():
                    batch_item[tensor_key] = Variable(batch_item[tensor_key].cuda())
                else:
                    batch_item[tensor_key] = Variable(batch_item[tensor_key])
        attn_matrix = extract_rnn_softmax_weights(model, batch_item, config, gpu_list)
        metrics_dict = compute_attention_metrics(attn_matrix, k_value=k_value)
        for b_idx, sample_guid in enumerate(batch_item["guid"]):
            record = extract_sample_record(
                sample_guid, metrics_dict, b_idx, gt_positives
            )
            collected_rows.append(record)
    return collected_rows


def process_single_divergent_evaluation(
    variant: str,
    model_name: str,
    divergent_pairs: list[tuple[str, str]],
    output_dir: str,
    experiment_ver: str,
    gpu_list: list[int],
    path_prefix: str,
    gt_positives: set[str] | None,
    k_value: int,
    generate_plots: bool,
) -> None:
    """Initialize model for a variant and model, filter dataset, and save outputs."""
    from config_parser.parser import create_config
    from tools.init_tool import init_all

    config_path = f"config/nlp/divergent/{variant.lower()}_{model_name.lower()}.config"
    metrics_path = resolve_validation_metrics_file(variant, model_name, experiment_ver)
    checkpoint_path = resolve_best_checkpoint_file(metrics_path, path_prefix)
    config = create_config(config_path)
    print(f"[*] Variant: {variant}, Model: {model_name}, Ckpt: {checkpoint_path}")
    parameters = init_all(config, gpu_list, checkpoint_path, "test")
    model = parameters["model"]
    dataloader = parameters["test_dataset"]
    filtered_loader = create_filtered_dataloader(dataloader, divergent_pairs)
    records = evaluate_divergent_dataloader(
        model, filtered_loader, config, gpu_list, gt_positives, k_value
    )
    os.makedirs(output_dir, exist_ok=True)
    export_pair_records_csv(records, os.path.join(output_dir, "pair_stats.csv"))
    export_aggregate_metrics_json(
        records, os.path.join(output_dir, "aggregate_stats.json"), k_value
    )
    if generate_plots:
        export_distribution_plots(
            records, os.path.join(output_dir, "histograms"), k_value
        )


def run_intra_divergent_pipeline(
    experiment_ver: str,
    output_base_dir: str,
    gpu_list: list[int],
    path_prefix: str,
    gt_positives: set[str] | None,
    k_value: int,
    generate_plots: bool,
) -> None:
    """Run divergent attention evaluation across intra-model segmentation variants."""
    models_evaluated = ["lstm", "gru"]
    intra_predictions = calculate_intra_divergent_pairs(
        experiment_ver, models_evaluated
    )
    for model_name in models_evaluated:
        for comp_title, pairs in intra_predictions[model_name].items():
            variant1, variant2 = comp_title.split(" vs ")
            for variant in [variant1, variant2]:
                folder_name = f"{variant.lower()}_{model_name.lower()}"
                dest_dir = f"{output_base_dir}/{experiment_ver}/intra/{folder_name}"
                process_single_divergent_evaluation(
                    variant=variant,
                    model_name=model_name,
                    divergent_pairs=pairs,
                    output_dir=dest_dir,
                    experiment_ver=experiment_ver,
                    gpu_list=gpu_list,
                    path_prefix=path_prefix,
                    gt_positives=gt_positives,
                    k_value=k_value,
                    generate_plots=generate_plots,
                )


def run_inter_divergent_pipeline(
    experiment_ver: str,
    output_base_dir: str,
    gpu_list: list[int],
    path_prefix: str,
    gt_positives: set[str] | None,
    k_value: int,
    generate_plots: bool,
) -> None:
    """Run divergent attention evaluation across inter-model architecture variants."""
    variants_evaluated = ["vanilla", "summarized",]
    inter_predictions = calculate_inter_divergent_pairs(
        experiment_ver, variants_evaluated
    )
    for variant in variants_evaluated:
        for comp_title, pairs in inter_predictions[variant].items():
            model1, model2 = comp_title.split(" vs ")
            for model_name in [model1, model2]:
                folder_name = f"{variant.lower()}_{model_name.lower()}"
                dest_dir = f"{output_base_dir}/{experiment_ver}/inter/{folder_name}"
                process_single_divergent_evaluation(
                    variant=variant,
                    model_name=model_name,
                    divergent_pairs=pairs,
                    output_dir=dest_dir,
                    experiment_ver=experiment_ver,
                    gpu_list=gpu_list,
                    path_prefix=path_prefix,
                    gt_positives=gt_positives,
                    k_value=k_value,
                    generate_plots=generate_plots,
                )


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for attention summarization."""
    parser = argparse.ArgumentParser(
        description="Attention weight summarizer for divergent model predictions."
    )
    parser.add_argument(
        "--experiment-version",
        "-ev",
        default="v1",
        help="Experiment version (e.g. v1, v2, v3)",
    )
    parser.add_argument(
        "--type",
        "-t",
        default="intra",
        choices=["intra", "inter"],
        help="Divergence type: 'intra' (between variants) or 'inter' (between models)",
    )
    parser.add_argument("--gpu", default="0", help="GPU id (default 0)")
    parser.add_argument(
        "--top_k", type=int, default=3, help="Top-K count for concentration metric"
    )
    parser.add_argument(
        "--ground_truth",
        default="data/COLIEE/task1_test_labels_2024.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--output_dir",
        default="output/results/divergent",
        help="Base output directory for divergent results",
    )
    parser.add_argument(
        "--path_prefix",
        default="/app",
        help="Prefix to strip from Docker paths in metrics JSON",
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate histogram PNG plots"
    )
    return parser


def main() -> None:
    """Execute divergent attention summarization CLI."""
    parser = build_cli_parser()
    args = parser.parse_args()
    gpu_list = [int(item) for item in args.gpu.split(",") if item.strip()]
    gt_positives = load_ground_truth_label_set(args.ground_truth)
    if args.type == "intra":
        run_intra_divergent_pipeline(
            experiment_ver=args.experiment_version,
            output_base_dir=args.output_dir,
            gpu_list=gpu_list,
            path_prefix=args.path_prefix,
            gt_positives=gt_positives,
            k_value=args.top_k,
            generate_plots=args.plots,
        )
    else:
        run_inter_divergent_pipeline(
            experiment_ver=args.experiment_version,
            output_base_dir=args.output_dir,
            gpu_list=gpu_list,
            path_prefix=args.path_prefix,
            gt_positives=gt_positives,
            k_value=args.top_k,
            generate_plots=args.plots,
        )


if __name__ == "__main__":
    main()
