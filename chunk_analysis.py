import json
import statistics
import os

def load_predictions(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_word_length(text):
    return len(text.split())

def analyze_dataset(data_path, pred_path):
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return None

    preds = load_predictions(pred_path)

    metrics = {
        'TP': {'q_lens': [], 'c_lens': [], 'comb_lens': []},
        'FP': {'q_lens': [], 'c_lens': [], 'comb_lens': []},
        'FN': {'q_lens': [], 'c_lens': [], 'comb_lens': []}
    }

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            guid = item['guid']
            q_id = guid.split('_')[0] + '.txt'
            c_id = guid.split('_')[1] + '.txt'
            true_label = item['label']
            
            pred_label = 1 if c_id in preds.get(q_id, []) else 0

            cat = None
            if true_label == 1 and pred_label == 1:
                cat = 'TP'
            elif true_label == 0 and pred_label == 1:
                cat = 'FP'
            elif true_label == 1 and pred_label == 0:
                cat = 'FN'
            
            if cat is not None:
                q_paras = item.get('q_paras', [])
                c_paras = item.get('c_paras', [])
                
                q_words = sum(calculate_word_length(c) for c in q_paras)
                c_words = sum(calculate_word_length(c) for c in c_paras)
                
                q_avg = q_words / len(q_paras) if q_paras else 0
                c_avg = c_words / len(c_paras) if c_paras else 0
                total_chunks = len(q_paras) + len(c_paras)
                comb_avg = (q_words + c_words) / total_chunks if total_chunks > 0 else 0
                
                metrics[cat]['q_lens'].append(q_avg)
                metrics[cat]['c_lens'].append(c_avg)
                metrics[cat]['comb_lens'].append(comb_avg)
                
    results = {}
    for cat in ['TP', 'FP', 'FN']:
        q_arr = metrics[cat]['q_lens']
        c_arr = metrics[cat]['c_lens']
        comb_arr = metrics[cat]['comb_lens']
        results[cat] = {
            'count': len(q_arr),
            'q_avg': sum(q_arr)/len(q_arr) if q_arr else 0,
            'c_avg': sum(c_arr)/len(c_arr) if c_arr else 0,
            'comb_avg': sum(comb_arr)/len(comb_arr) if comb_arr else 0
        }
        
    return results

def print_markdown_table(all_results):
    print("### Chunk Length Analysis (Average Words per Chunk)")
    print()
    print("| Approach | Outcome | Count | Query Avg Words | Candidate Avg Words | Combined Avg Words |")
    print("|---|---|---|---|---|---|")
    
    for approach_name, results in all_results.items():
        if results is None:
            continue
        for cat in ['TP', 'FP', 'FN']:
            res = results[cat]
            print(f"| {approach_name} | {cat} | {res['count']} | {res['q_avg']:.2f} | {res['c_avg']:.2f} | {res['comb_avg']:.2f} |")

def main():
    approaches = [
        ("Vanilla Sentences", "data/COLIEE/test_vanilla_sentences.json", "output/results/vanilla/non_poolout/lstm_parsed_results.json"),
        ("Summarized Sentences", "data/COLIEE/test_summarized_sentences.json", "output/results/summarized/non_poolout/lstm_parsed_results.json"),
        ("Vanilla Paragraphs", "data/COLIEE/test_vanilla_paragraphs.json", "output/results/paragraph/non_poolout/lstm_parsed_results.json")
    ]
    
    all_results = {}
    for name, data_path, pred_path in approaches:
        res = analyze_dataset(data_path, pred_path)
        all_results[name] = res
        
    print_markdown_table(all_results)

if __name__ == "__main__":
    main()
