import argparse
import json
import os
from collections import defaultdict


def load_json(filepath, report=None):
    if not os.path.exists(filepath):
        message = f"Warning: File {filepath} does not exist."
        if report is not None:
            report.emit(message)
        else:
            print(message)
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_metrics(predictions, ground_truth):
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    total_tp, total_fp, total_fn = 0, 0, 0

    num_queries = len(ground_truth)
    if num_queries == 0:
        return {}

    for query, gt_docs in ground_truth.items():
        pred_docs = predictions.get(query, [])

        gt_set = set(gt_docs)
        pred_set = set(pred_docs)

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        macro_p += p
        macro_r += r
        macro_f1 += f1

    macro_p /= num_queries
    macro_r /= num_queries
    macro_f1 /= num_queries

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1},
        'micro': {'precision': micro_p, 'recall': micro_r, 'f1': micro_f1},
        'totals': {'tp': total_tp, 'fp': total_fp, 'fn': total_fn}
    }


def calculate_agreement(preds1, preds2):
    total_jaccard = 0.0
    common_queries = set(preds1.keys()) & set(preds2.keys())

    if not common_queries:
        return 0.0

    for query in common_queries:
        set1 = set(preds1[query])
        set2 = set(preds2[query])

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union > 0:
            total_jaccard += intersection / union
        else:
            total_jaccard += 1.0

    return total_jaccard / len(common_queries)


def calculate_overlap_counts(models_dict):
    common_queries = set.intersection(*(set(m.keys()) for m in models_dict.values()))
    model_count = len(models_dict)

    doc_sources = defaultdict(lambda: defaultdict(list))

    for model_name, preds in models_dict.items():
        for query in common_queries:
            for doc in preds[query]:
                doc_sources[query][doc].append(model_name)

    agreement_counts = {
        'all_models': 0,
        'unique_to_1': 0,
    }

    if model_count >= 3:
        agreement_counts['exactly_2'] = 0

    for docs in doc_sources.values():
        for sources in docs.values():
            if len(sources) == model_count:
                agreement_counts['all_models'] += 1
            elif len(sources) == 1:
                agreement_counts['unique_to_1'] += 1
            elif model_count >= 3 and len(sources) == 2:
                agreement_counts['exactly_2'] += 1

    return agreement_counts


class MarkdownReport:
    def __init__(self):
        self.lines = []

    def emit(self, text=""):
        print(text)
        self.lines.append(text)

    def heading(self, level, text):
        self.emit(f"{'#' * level} {text}")
        self.emit("")

    def table(self, headers, rows):
        self.emit("| " + " | ".join(headers) + " |")
        self.emit("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            self.emit("| " + " | ".join(row) + " |")
        self.emit("")

    def write(self, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("\n".join(self.lines).rstrip() + "\n")


def format_score(value):
    return f"{value:.4f}"


def format_percent(value):
    return f"{value:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Compare model results and write a Markdown report.")
    parser.add_argument(
        "--output",
        default="output/results/v2/compare_results_report.md",
        help="Path to the Markdown report to create.",
    )
    args = parser.parse_args()

    report = MarkdownReport()
    gt_path = 'data/COLIEE/task1_test_labels_2024.json'
    gt = load_json(gt_path, report)

    report.heading(1, 'Compare vanilla/summarized to test labels')
    report.emit(f"- Ground truth: `{gt_path}`")
    report.emit(f"- Report output: `{args.output}`")
    report.emit("")

    if not gt:
        report.emit("Ground truth data could not be loaded, so no metrics were computed.")
        report.write(args.output)
        return

    for model in ['lstm', 'gru', 'transformer']:
        report.heading(2, f"Evaluating Model: {model.upper()}")

        vanilla_path = f'output/results/v2/vanilla/{model}_parsed_results.json'
        summarized_path = f'output/results/v2/summarized/{model}_parsed_results.json'
        # paragraph_path = f'output/results/paragraph/{model}_parsed_results.json'
        paragraph_path = None

        report.emit('Loading data...')
        vanilla = load_json(vanilla_path, report)
        summarized = load_json(summarized_path, report)

        models = {
            'Vanilla': vanilla,
            'Summarized': summarized
        }

        if model != 'transformer' and paragraph_path is not None:
            paragraph = load_json(paragraph_path, report)
            models['Paragraph'] = paragraph

        report.heading(3, 'Performance vs Ground Truth')
        for name, preds in models.items():
            if not preds:
                report.emit(f"- {name}: No predictions loaded.")
                report.emit("")
                continue

            metrics = calculate_metrics(preds, gt)
            if not metrics:
                report.emit(f"- {name}: Ground truth is empty, metrics were not computed.")
                report.emit("")
                continue

            report.heading(4, f"{name} Results")
            report.table(
                ['Scope', 'Precision', 'Recall', 'F1'],
                [
                    ['Micro', format_score(metrics['micro']['precision']), format_score(metrics['micro']['recall']), format_score(metrics['micro']['f1'])],
                    ['Macro', format_score(metrics['macro']['precision']), format_score(metrics['macro']['recall']), format_score(metrics['macro']['f1'])],
                ],
            )
            report.table(
                ['Metric', 'Value'],
                [
                    ['TP', str(metrics['totals']['tp'])],
                    ['FP', str(metrics['totals']['fp'])],
                    ['FN', str(metrics['totals']['fn'])],
                ],
            )

        report.heading(3, 'Model Agreement (Jaccard Similarity)')
        model_names = list(models.keys())
        agreement_rows = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]
                if models[m1] and models[m2]:
                    agreement = calculate_agreement(models[m1], models[m2])
                    agreement_rows.append([m1, m2, format_score(agreement)])

        if agreement_rows:
            report.table(['Model A', 'Model B', 'Jaccard'], agreement_rows)
        else:
            report.emit('- No agreement pairs could be computed.')
            report.emit('')

        report.heading(3, 'Prediction Overlap Analysis')
        valid_models = {k: v for k, v in models.items() if v}
        if len(valid_models) >= 2:
            counts = calculate_overlap_counts(valid_models)
            total_unique_preds = sum(counts.values())
            if total_unique_preds > 0:
                overlap_rows = [
                    ['Predicted by all models', str(counts['all_models']), format_percent(counts['all_models'] / total_unique_preds * 100)],
                ]
                if 'exactly_2' in counts:
                    overlap_rows.append([
                        'Predicted by exactly 2 models',
                        str(counts['exactly_2']),
                        format_percent(counts['exactly_2'] / total_unique_preds * 100),
                    ])
                overlap_rows.append([
                    'Predicted by only 1 model',
                    str(counts['unique_to_1']),
                    format_percent(counts['unique_to_1'] / total_unique_preds * 100),
                ])

                report.emit(f'- Total unique (query, doc) predictions across all models: {total_unique_preds}')
                report.table(['Category', 'Count', 'Percent'], overlap_rows)
        else:
            report.emit('- Not enough non-empty model outputs to compute overlap.')
            report.emit('')

    report.heading(1, 'Evaluation Complete')
    report.write(args.output)


if __name__ == '__main__':
    main()
