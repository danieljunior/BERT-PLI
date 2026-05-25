import json
import os
from collections import defaultdict

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist.")
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
    """Calculate Jaccard similarity / agreement between two sets of predictions."""
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
            total_jaccard += 1.0 # Both predicted empty
            
    return total_jaccard / len(common_queries)

def calculate_overlap_counts(models_dict):
    """Calculate how many predictions are identical across all 3, any 2, or unique."""
    # Find all queries common to all models
    common_queries = set.intersection(*(set(m.keys()) for m in models_dict.values()))
    
    all_3_agreed = 0
    total_predictions = 0
    
    # Track which models predicted which documents
    # {query: {doc: [model1, model2]}}
    doc_sources = defaultdict(lambda: defaultdict(list))
    
    for model_name, preds in models_dict.items():
        for query in common_queries:
            for doc in preds[query]:
                doc_sources[query][doc].append(model_name)
                total_predictions += 1
                
    agreement_counts = {
        'all_3': 0,
        'exactly_2': 0,
        'unique_to_1': 0
    }
    
    for query, docs in doc_sources.items():
        for doc, sources in docs.items():
            if len(sources) == 3:
                agreement_counts['all_3'] += 1
            elif len(sources) == 2:
                agreement_counts['exactly_2'] += 1
            else:
                agreement_counts['unique_to_1'] += 1
                
    # Since all_3 counts a doc that 3 models predicted, we should adjust if we want relative percentages
    # Let's just return the raw counts of unique (Query, Doc) pairs.
    return agreement_counts

def main():
    gt_path = 'data/COLIEE/task1_test_labels_2024.json'
    vanilla_path = 'output/results/vanilla/non_poolout/lstm_parsed_results.json'
    summarized_path = 'output/results/summarized/non_poolout/lstm_parsed_results.json'
    paragraph_path = 'output/results/paragraph/non_poolout/lstm_parsed_results.json'
    
    print("Loading data...")
    gt = load_json(gt_path)
    vanilla = load_json(vanilla_path)
    summarized = load_json(summarized_path)
    paragraph = load_json(paragraph_path)
    
    models = {
        'Vanilla': vanilla,
        'Summarized': summarized,
        'Paragraph': paragraph
    }
    
    print("\n--- Performance vs Ground Truth ---")
    for name, preds in models.items():
        if not preds:
            print(f"{name}: No predictions loaded.")
            continue
            
        metrics = calculate_metrics(preds, gt)
        print(f"\n{name} Results:")
        print(f"  Micro - Precision: {metrics['micro']['precision']:.4f}, Recall: {metrics['micro']['recall']:.4f}, F1: {metrics['micro']['f1']:.4f}")
        print(f"  Macro - Precision: {metrics['macro']['precision']:.4f}, Recall: {metrics['macro']['recall']:.4f}, F1: {metrics['macro']['f1']:.4f}")
        print(f"  Totals - TP: {metrics['totals']['tp']}, FP: {metrics['totals']['fp']}, FN: {metrics['totals']['fn']}")
        
    print("\n--- Model Agreement (Jaccard Similarity) ---")
    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            if models[m1] and models[m2]:
                agreement = calculate_agreement(models[m1], models[m2])
                print(f"  {m1} vs {m2}: {agreement:.4f}")

    print("\n--- Prediction Overlap Analysis ---")
    valid_models = {k: v for k, v in models.items() if v}
    if len(valid_models) == 3:
        counts = calculate_overlap_counts(valid_models)
        total_unique_preds = sum(counts.values())
        if total_unique_preds > 0:
            print(f"  Total Unique (Query, Doc) Predictions across all models: {total_unique_preds}")
            print(f"  Predicted by ALL 3 models: {counts['all_3']} ({counts['all_3']/total_unique_preds*100:.1f}%)")
            print(f"  Predicted by EXACTLY 2 models: {counts['exactly_2']} ({counts['exactly_2']/total_unique_preds*100:.1f}%)")
            print(f"  Predicted by ONLY 1 model: {counts['unique_to_1']} ({counts['unique_to_1']/total_unique_preds*100:.1f}%)")

if __name__ == '__main__':
    main()
