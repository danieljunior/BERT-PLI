"""
Loader and filter utilities for divergent model prediction subsets.

Calculates symmetric difference pairs between segmentation variants (intra)
and model architectures (inter).
"""

import json
import os
from pathlib import Path


def load_predictions_json(filepath: str) -> dict[str, list[str]]:
    """Load JSON file mapping query document names to retrieved candidates."""
    target_path = Path(filepath)
    if not target_path.is_file():
        msg = f"Predictions file not found: '{filepath}'. Expected existing JSON."
        raise FileNotFoundError(msg)
    with target_path.open("r", encoding="utf-8") as file_handle:
        parsed_data = json.load(file_handle)
    if not isinstance(parsed_data, dict):
        msg = f"Predictions in '{filepath}' must be a dict, got {type(parsed_data)}"
        raise TypeError(msg)
    return parsed_data


def load_intra_predictions(
    model_name: str, experiment_ver: str, base_dir: str = "output/results"
) -> dict[str, dict[str, list[str]]]:
    """Load vanilla and summarized parsed results for intra-model comparisons."""
    vanilla_path = f"{base_dir}/vanilla/{experiment_ver}_{model_name}_parsed_results.json"
    sumy_path = f"{base_dir}/summarized/{experiment_ver}_{model_name}_parsed_results.json"
    vanilla_preds = load_predictions_json(vanilla_path)
    sumy_preds = load_predictions_json(sumy_path)
    return {"Vanilla": vanilla_preds, "Summarized": sumy_preds}


def load_inter_predictions(
    variant_name: str, experiment_ver: str, base_dir: str = "output/results"
) -> dict[str, dict[str, list[str]]]:
    """Load GRU and LSTM parsed results for inter-model comparisons."""
    gru_path = f"{base_dir}/{variant_name}/{experiment_ver}_gru_parsed_results.json"
    lstm_path = f"{base_dir}/{variant_name}/{experiment_ver}_lstm_parsed_results.json"
    gru_preds = load_predictions_json(gru_path)
    lstm_preds = load_predictions_json(lstm_path)
    return {"GRU": gru_preds, "LSTM": lstm_preds}


def extract_symmetric_difference(
    first_preds: dict[str, list[str]], second_preds: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """Find query-candidate pairs present in one prediction set but not both."""
    shared_queries = set(first_preds.keys()) & set(second_preds.keys())
    if not shared_queries:
        return []
    divergent_pairs: list[tuple[str, str]] = []
    for query in sorted(shared_queries):
        first_set = set(first_preds[query])
        second_set = set(second_preds[query])
        symm_diff = first_set ^ second_set
        for candidate in sorted(symm_diff):
            divergent_pairs.append((query, candidate))
    return divergent_pairs


def calculate_intra_divergent_pairs(
    experiment_ver: str, models: list[str] | None = None
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Compute divergent pairs between segmentation variants for each model."""
    evaluated_models = models or ["lstm", "gru"]
    intra_results: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for model in evaluated_models:
        variant_preds = load_intra_predictions(model, experiment_ver)
        intra_results[model] = {}
        comparison_name = "Vanilla vs Summarized"
        pairs = extract_symmetric_difference(
            variant_preds["Vanilla"], variant_preds["Summarized"]
        )
        intra_results[model][comparison_name] = pairs
    return intra_results


def calculate_inter_divergent_pairs(
    experiment_ver: str, variants: list[str] | None = None
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Compute divergent pairs between GRU and LSTM models for each variant."""
    evaluated_variants = variants or ["vanilla", "summarized"]
    inter_results: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for variant in evaluated_variants:
        model_preds = load_inter_predictions(variant, experiment_ver)
        inter_results[variant] = {}
        comparison_name = "GRU vs LSTM"
        pairs = extract_symmetric_difference(model_preds["GRU"], model_preds["LSTM"])
        inter_results[variant][comparison_name] = pairs
    return inter_results


def extract_target_guid_set(divergent_pairs: list[tuple[str, str]]) -> set[str]:
    """Convert (query, candidate) tuples to normalized guid strings '{q}_{c}'."""
    guid_set: set[str] = set()
    for query, candidate in divergent_pairs:
        clean_q = query[:-4] if query.endswith(".txt") else query
        clean_c = candidate[:-4] if candidate.endswith(".txt") else candidate
        guid_set.add(f"{clean_q}_{clean_c}")
    return guid_set


def collect_filtered_indices(
    dataset_items: list[dict[str, str]], target_guids: set[str]
) -> list[int]:
    """Collect dataset indices matching the target guid set in O(1) per item."""
    matching_indices: list[int] = []
    for idx, item in enumerate(dataset_items):
        item_guid = item.get("guid", "")
        if item_guid in target_guids:
            matching_indices.append(idx)
    return matching_indices
