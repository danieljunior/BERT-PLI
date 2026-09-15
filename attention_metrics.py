"""
Pure numerical metrics for attention weight distributions.

Calculates Shannon entropy, top-K concentration sums, and Gini inequality coefficients.
"""

from collections import Counter
import numpy as np


def compute_sample_entropy(weights: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Compute Shannon entropy per row for probability distributions."""
    if weights.ndim != 2:
        msg = f"Expected 2D array [batch, seq], got shape {weights.shape}"
        raise ValueError(msg)
    return -np.sum(weights * np.log(weights + epsilon), axis=-1)


def compute_topk_concentration(weights: np.ndarray, k_value: int) -> np.ndarray:
    """Sum the top-K highest attention weights per row."""
    if weights.ndim != 2:
        msg = f"Expected 2D array [batch, seq], got shape {weights.shape}"
        raise ValueError(msg)
    clamped_k = min(k_value, weights.shape[1])
    sorted_weights = np.sort(weights, axis=-1)[:, ::-1]
    return sorted_weights[:, :clamped_k].sum(axis=-1)


def compute_gini_coefficient(weights: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Compute Gini inequality index per row across attention weights."""
    if weights.ndim != 2:
        msg = f"Expected 2D array [batch, seq], got shape {weights.shape}"
        raise ValueError(msg)
    num_elements = weights.shape[1]
    sorted_asc = np.sort(weights, axis=-1)
    indices = np.arange(1, num_elements + 1)
    weight_totals = sorted_asc.sum(axis=-1)
    numerator = 2.0 * np.sum(indices * sorted_asc, axis=-1)
    denominator = num_elements * weight_totals + epsilon
    raw_gini = (numerator / denominator) - (num_elements + 1.0) / num_elements
    return np.clip(raw_gini, 0.0, 1.0)


def compute_attention_metrics(weights: np.ndarray, k_value: int = 3) -> dict[str, np.ndarray]:
    """Calculate dictionary of attention distribution metrics per sample."""
    if weights.ndim != 2:
        msg = f"Expected 2D array [batch, seq], got shape {weights.shape}"
        raise ValueError(msg)
    entropy_values = compute_sample_entropy(weights)
    max_values = weights.max(axis=-1)
    argmax_indices = weights.argmax(axis=-1)
    min_values = weights.min(axis=-1)
    topk_values = compute_topk_concentration(weights, k_value)
    gini_values = compute_gini_coefficient(weights)
    return {
        "entropy": entropy_values,
        "max_weight": max_values,
        "argmax_weight": argmax_indices,
        "min_weight": min_values,
        f"top{k_value}_concentration": topk_values,
        "gini": gini_values,
    }


def compute_numeric_summary(values: np.ndarray) -> dict[str, float]:
    """Calculate summary statistics for a 1D array of metric values."""
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_frequency_counts(int_values: list[int]) -> dict[str, int]:
    """Count occurrence frequencies of integer categorical values."""
    counts = Counter(int_values)
    return {str(key): count for key, count in sorted(counts.items())}
