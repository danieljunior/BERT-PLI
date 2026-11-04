"""
Unit tests for divergent_entropy_to_latex.py
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from divergent_entropy_to_latex import (
    load_divergent_entropy_stats,
    generate_grouped_latex_table,
    ORDERED_VARIANTS,
)


class TestDivergentEntropyToLatex(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.base_path = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_stats(
        self, version: str, variant: str, mean: float, std: float, samples: int
    ) -> None:
        var_dir = self.base_path / version / "inter" / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        stats_file = var_dir / "aggregate_stats.json"
        data = {
            "total_samples": samples,
            "entropy": {
                "mean": mean,
                "std": std,
            },
        }
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_load_divergent_entropy_stats(self) -> None:
        self._create_mock_stats("v1", "summarized_gru", 2.7812, 0.5935, 599)
        self._create_mock_stats("v1", "vanilla_lstm", 0.3052, 0.3755, 1115)

        stats = load_divergent_entropy_stats(str(self.base_path))
        self.assertIn("v1", stats)
        self.assertIn("summarized_gru", stats["v1"])
        self.assertIn("vanilla_lstm", stats["v1"])

        self.assertAlmostEqual(stats["v1"]["summarized_gru"]["mean"], 2.7812)
        self.assertAlmostEqual(stats["v1"]["summarized_gru"]["std"], 0.5935)
        self.assertEqual(stats["v1"]["summarized_gru"]["total_samples"], 599)

    def test_generate_grouped_latex_table_format(self) -> None:
        stats = {
            "v1": {
                "summarized_gru": {"mean": 2.7812, "std": 0.5935, "total_samples": 599},
                "vanilla_lstm": {"mean": 0.3052, "std": 0.3755, "total_samples": 1115},
            }
        }
        table_tex = generate_grouped_latex_table(stats, precision=2)

        self.assertIn(r"\begin{table}", table_tex)
        self.assertIn(r"\begin{tabular}{llcc}", table_tex)
        self.assertIn(r"\toprule", table_tex)
        self.assertIn(r"\midrule", table_tex)
        self.assertIn(r"\bottomrule", table_tex)
        self.assertIn(r"\end{tabular}", table_tex)
        self.assertIn(r"\end{table}", table_tex)

        # 2 decimal places check
        self.assertIn(r"$2.78 \pm 0.59$", table_tex)
        self.assertIn(r"$0.31 \pm 0.38$", table_tex)
        self.assertIn(r"599", table_tex)
        self.assertIn(r"1115", table_tex)
        self.assertIn(r"\multirow{2}{*}{$v1$}", table_tex)

    def test_nonexistent_base_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_divergent_entropy_stats("/path/that/does/not/exist")


if __name__ == "__main__":
    unittest.main()
