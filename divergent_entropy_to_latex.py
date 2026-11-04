#!/usr/bin/env python3
"""
Generate LaTeX tables of mean attention entropy and standard deviation
for inter-model divergent predictions across versions (v1, v2, v3).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


ORDERED_VARIANTS: List[str] = [
    "summarized_gru",
    "summarized_lstm",
    "vanilla_gru",
    "vanilla_lstm",
]

VARIANT_DISPLAY_NAMES: Dict[str, str] = {
    "summarized_gru": "Summarized GRU",
    "summarized_lstm": "Summarized LSTM",
    "vanilla_gru": "Vanilla GRU",
    "vanilla_lstm": "Vanilla LSTM",
}


def load_divergent_entropy_stats(
    base_dir: str = "output/results/divergent",
    versions: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load entropy mean, std, and total_samples from aggregate_stats.json files.

    Expected directory structure:
        {base_dir}/{version}/inter/{variant}/aggregate_stats.json

    Returns:
        Nested dict: {version: {variant: {'mean': float, 'std': float, 'total_samples': int}}}
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    if versions is None:
        detected_versions = []
        for p in sorted(base_path.iterdir()):
            if p.is_dir() and (p / "inter").is_dir():
                detected_versions.append(p.name)
        versions = detected_versions

    stats_by_version: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for version in versions:
        inter_dir = base_path / version / "inter"
        if not inter_dir.is_dir():
            continue

        variant_stats: Dict[str, Dict[str, Any]] = {}
        for variant in ORDERED_VARIANTS:
            stats_file = inter_dir / variant / "aggregate_stats.json"
            if not stats_file.is_file():
                continue

            with stats_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            entropy_data = data.get("entropy", {})
            mean_val = entropy_data.get("mean")
            std_val = entropy_data.get("std")
            total_samples = data.get("total_samples")

            if mean_val is not None and std_val is not None:
                variant_stats[variant] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "total_samples": int(total_samples) if total_samples is not None else 0,
                }

        if variant_stats:
            stats_by_version[version] = variant_stats

    return stats_by_version


def generate_grouped_latex_table(
    stats_by_version: Dict[str, Dict[str, Dict[str, Any]]],
    precision: int = 2,
    caption: str = "Média e Desvio Padrão da Entropia de Atenção em Predições Divergentes (Inter-Modelo)",
    label: str = "tab:divergent_entropy_grouped",
) -> str:
    """
    Format statistics as a single LaTeX table with subdivisions for each version.
    """
    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Versão} & \textbf{Configuração} & \textbf{Entropia ($\text{Média} \pm \text{DP}$)} & \textbf{Pares Divergentes ($N$)} \\",
    ]

    version_keys = sorted(stats_by_version.keys())

    for idx, version in enumerate(version_keys):
        variants = stats_by_version[version]
        lines.append(r"\midrule")
        num_variants = len(variants)
        version_label = f"${version}$"

        is_first = True
        for variant_key in ORDERED_VARIANTS:
            if variant_key not in variants:
                continue

            v_info = variants[variant_key]
            mean_str = f"{v_info['mean']:.{precision}f}"
            std_str = f"{v_info['std']:.{precision}f}"
            entropy_fmt = f"${mean_str} \\pm {std_str}$"
            samples_fmt = str(v_info["total_samples"])
            display_name = VARIANT_DISPLAY_NAMES.get(variant_key, variant_key)

            if is_first:
                version_col = f"\\multirow{{{num_variants}}}{{*}}{{{version_label}}}"
                lines.append(f"{version_col} & {display_name} & {entropy_fmt} & {samples_fmt} \\\\")
                is_first = False
            else:
                lines.append(f" & {display_name} & {entropy_fmt} & {samples_fmt} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table of attention entropy for divergent predictions."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="output/results/divergent",
        help="Base path containing divergent results (default: output/results/divergent)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal precision for mean and std (default: 2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/results/divergent/divergent_entropy_table.tex",
        help="Output path for .tex file (default: output/results/divergent/divergent_entropy_table.tex)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print LaTeX table to stdout",
    )

    args = parser.parse_args()

    stats = load_divergent_entropy_stats(args.results_dir)
    latex_code = generate_grouped_latex_table(stats, precision=args.precision)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(latex_code + "\n")
        print(f"[+] LaTeX table successfully exported to {out_path}")

    if not args.quiet:
        print("\n" + latex_code + "\n")


if __name__ == "__main__":
    main()
