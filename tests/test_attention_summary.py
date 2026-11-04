"""
Unit tests for attention_summary, attention_metrics, and divergent_subset_loader.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from attention_metrics import (
    compute_sample_entropy,
    compute_topk_concentration,
    compute_gini_coefficient,
    compute_attention_metrics,
    compute_numeric_summary,
    summarize_frequency_counts,
)
from divergent_subset_loader import (
    extract_symmetric_difference,
    extract_target_guid_set,
    collect_filtered_indices,
    calculate_intra_divergent_pairs,
    calculate_inter_divergent_pairs,
)
from attention_summary import (
    clean_identifier_stem,
    load_ground_truth_label_set,
    export_pair_records_csv,
    load_pair_records_csv,
    export_aggregate_metrics_json,
    build_label_breakdown,
)


class TestAttentionMetrics(unittest.TestCase):
    """Tests for pure numerical attention metrics."""

    def test_entropy_uniform_vs_peaked(self) -> None:
        """Uniform distribution should have higher entropy than peaked."""
        uniform_w = np.full((1, 4), 0.25)
        peaked_w = np.array([[0.97, 0.01, 0.01, 0.01]])
        h_uniform = compute_sample_entropy(uniform_w)
        h_peaked = compute_sample_entropy(peaked_w)
        self.assertGreater(h_uniform[0], h_peaked[0])
        self.assertAlmostEqual(h_uniform[0], np.log(4), places=4)

    def test_topk_concentration_bounds(self) -> None:
        """Top-K concentration must sum top K elements within [0, 1]."""
        weights = np.array([[0.4, 0.3, 0.2, 0.1]])
        top2 = compute_topk_concentration(weights, k_value=2)
        self.assertAlmostEqual(top2[0], 0.7, places=5)

    def test_gini_coefficient_bounds(self) -> None:
        """Gini should be near 0 for uniform and near 1 for peaked distributions."""
        uniform_w = np.full((1, 10), 0.1)
        peaked_w = np.zeros((1, 10))
        peaked_w[0, 0] = 1.0
        g_uniform = compute_gini_coefficient(uniform_w)
        g_peaked = compute_gini_coefficient(peaked_w)
        self.assertAlmostEqual(g_uniform[0], 0.0, places=3)
        self.assertGreater(g_peaked[0], 0.8)

    def test_compute_attention_metrics_keys(self) -> None:
        """Full metrics dictionary must contain all expected metric keys."""
        weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        res = compute_attention_metrics(weights, k_value=2)
        expected_keys = {
            "entropy",
            "max_weight",
            "argmax_weight",
            "min_weight",
            "top2_concentration",
            "gini",
        }
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertEqual(res["argmax_weight"].tolist(), [0, 1])

    def test_numeric_summary_aggregation(self) -> None:
        """Numeric summary computes correct mean, std, and percentiles."""
        data_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = compute_numeric_summary(data_arr)
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 5.0)


class TestDivergentSubsetLoader(unittest.TestCase):
    """Tests for divergent prediction extraction and index filtering."""

    def test_extract_symmetric_difference(self) -> None:
        """Symmetric difference should isolate discordant query-candidate pairs."""
        preds_a = {"q1": ["c1", "c2"], "q2": ["c3"]}
        preds_b = {"q1": ["c2", "c4"], "q2": ["c3"]}
        diff = extract_symmetric_difference(preds_a, preds_b)
        expected = [("q1", "c1"), ("q1", "c4")]
        self.assertEqual(sorted(diff), sorted(expected))

    def test_extract_target_guid_set(self) -> None:
        """Guid normalizer strips .txt and formats as '{q}_{c}'."""
        pairs = [("001.txt", "002.txt"), ("003", "004.txt")]
        guids = extract_target_guid_set(pairs)
        self.assertEqual(guids, {"001_002", "003_004"})

    def test_collect_filtered_indices(self) -> None:
        """Filters dataset items matching target guid in O(1) per lookup."""
        dataset = [
            {"guid": "001_002"},
            {"guid": "001_003"},
            {"guid": "002_005"},
        ]
        target = {"001_003", "002_005"}
        indices = collect_filtered_indices(dataset, target)
        self.assertEqual(indices, [1, 2])

    def test_calculate_intra_divergent_pairs_v1(self) -> None:
        """Verify intra divergent pairs can be loaded for v1 experiment."""
        intra = calculate_intra_divergent_pairs("v1", ["gru"])
        self.assertIn("gru", intra)
        self.assertIn("Vanilla vs Summarized", intra["gru"])
        pairs = intra["gru"]["Vanilla vs Summarized"]
        self.assertGreater(len(pairs), 0)

    def test_calculate_inter_divergent_pairs_v1(self) -> None:
        """Verify inter divergent pairs can be loaded for v1 experiment."""
        inter = calculate_inter_divergent_pairs("v1", ["vanilla"])
        self.assertIn("vanilla", inter)
        self.assertIn("GRU vs LSTM", inter["vanilla"])
        pairs = inter["vanilla"]["GRU vs LSTM"]
        self.assertGreater(len(pairs), 0)


class TestExporters(unittest.TestCase):
    """Tests for CSV and JSON output export routines."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_export_pair_records_csv(self) -> None:
        """CSV exporter writes correct headers and values."""
        records: list[dict[str, str | int | float]] = [
            {"guid": "q1_c1", "entropy": 1.23, "max_weight": 0.45, "gt_label": 1},
            {"guid": "q1_c2", "entropy": 2.34, "max_weight": 0.56, "gt_label": 0},
        ]
        target_csv = os.path.join(self.temp_dir, "test.csv")
        export_pair_records_csv(records, target_csv)
        self.assertTrue(os.path.isfile(target_csv))
        with open(target_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

    def test_export_aggregate_metrics_json(self) -> None:
        """JSON exporter aggregates metrics and breakdown by label."""
        records: list[dict[str, str | int | float]] = [
            {
                "guid": "q1_c1",
                "entropy": 1.0,
                "max_weight": 0.5,
                "min_weight": 0.1,
                "top2_concentration": 0.8,
                "gini": 0.4,
                "argmax_weight": 0,
                "gt_label": 1,
            },
            {
                "guid": "q1_c2",
                "entropy": 2.0,
                "max_weight": 0.6,
                "min_weight": 0.2,
                "top2_concentration": 0.9,
                "gini": 0.5,
                "argmax_weight": 1,
                "gt_label": 0,
            },
        ]
        target_json = os.path.join(self.temp_dir, "test.json")
        export_aggregate_metrics_json(records, target_json, k_value=2)
        self.assertTrue(os.path.isfile(target_json))
        with open(target_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_samples"], 2)
        self.assertIn("by_label", data)
        self.assertEqual(data["by_label"]["positive_count"], 1)

    def test_load_pair_records_csv_infers_top_k(self) -> None:
        """CSV loader converts values and infers the concentration metric K."""
        target_csv = os.path.join(self.temp_dir, "pair_stats.csv")
        with open(target_csv, "w", encoding="utf-8") as file_handle:
            file_handle.write(
                "guid,entropy,max_weight,argmax_weight,min_weight,"
                "top5_concentration,gini,gt_label\n"
                "q1_c1,1.2,0.5,3,0.1,0.8,0.4,1\n"
            )

        records, k_value = load_pair_records_csv(target_csv)

        self.assertEqual(k_value, 5)
        self.assertEqual(records[0]["argmax_weight"], 3)
        self.assertEqual(records[0]["gt_label"], 1)

    def test_load_pair_records_csv_rejects_missing_columns(self) -> None:
        """CSV loader reports malformed input rather than silently continuing."""
        target_csv = os.path.join(self.temp_dir, "invalid.csv")
        with open(target_csv, "w", encoding="utf-8") as file_handle:
            file_handle.write("guid,entropy\nq1_c1,1.2\n")

        with self.assertRaises(ValueError):
            load_pair_records_csv(target_csv)


if __name__ == "__main__":
    unittest.main()
