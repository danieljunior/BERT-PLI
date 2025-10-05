import json
from pathlib import Path
import random
from tqdm import tqdm
import spacy
import ray

random.seed(42)
spacy.cli.download("en_core_web_sm")

def read_text_file(file_path):
    """Read text file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_sentences(text):
    """Extract sentences using spaCy"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def generate_negative_pairs(labels, files):
    """Generate negative examples by randomly pairing documents"""
    negative_pairs = []
    all_docs = list(files)

    for q_file in tqdm(labels.keys(), desc="Generating negative pairs"):
        # Get all possible p_files excluding those that are positive pairs
        available_docs = [doc for doc in all_docs if doc not in labels[q_file]]
        
        # Generate same number of negative pairs as positive pairs
        num_negatives = len(labels[q_file])
        if available_docs:
            negative_pairs.extend([
                (q_file, random.choice(available_docs))
                for _ in range(num_negatives)
            ])
    
    return negative_pairs

def process_files(files_path, labels_file, output_file):
    # Load spaCy model
    # nlp = spacy.load("en_core_web_sm")
    
    # Read labels file
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Get all files in directory
    files = set(Path(files_path).glob('*.txt'))
    files = {f.name for f in files}
    
    # Generate positive and negative pairs
    positive_pairs = [(q, p) for q in labels for p in labels[q]]
    negative_pairs = generate_negative_pairs(labels, files)
    
    # Process all pairs
    output_data = []
    
    # Process positive pairs
    ray.init(num_cpus=6, ignore_reinit_error=True)

    @ray.remote
    def process_pair(q_file, p_file, files_path, positive_pairs_set):
        q_content = read_text_file(Path(files_path) / q_file)
        p_content = read_text_file(Path(files_path) / p_file)
        q_sentences = get_sentences(q_content)
        p_sentences = get_sentences(p_content)
        entry = {
            "guid": f"{q_file.split('.')[0]}_{p_file.split('.')[0]}",
            "q_paras": q_sentences,
            "c_paras": p_sentences,
            "label": 1 if (q_file, p_file) in positive_pairs_set else 0
        }
        return entry

    positive_pairs_set = set(positive_pairs)
    all_pairs = positive_pairs + negative_pairs

    # Ray cannot serialize spaCy models directly, so load it inside the remote function
    # We'll pass nlp=None and reload inside process_pair if needed
    # For efficiency, you can use ray's object store or actor, but for simplicity:
    def process_pair_wrapper(q_file, p_file, files_path, positive_pairs_set):
        nlp = spacy.load("en_core_web_sm")
        return process_pair.remote(q_file, p_file, files_path, positive_pairs_set)

    # futures = [
    #     process_pair_wrapper(q_file, p_file, files_path, positive_pairs_set)
    #     for q_file, p_file in tqdm(all_pairs, desc="Dispatching Ray tasks")
    #     ]
    # output_data = ray.get(futures)
    batch_size = 10
    output_data = []
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Batching Ray tasks"):
        batch = all_pairs[i:i + batch_size]
        futures = [
            process_pair_wrapper(q_file, p_file, files_path, positive_pairs_set)
            for q_file, p_file in tqdm(batch, desc="Dispatching Ray tasks", leave=False)
        ]
        output_data.extend(ray.get(futures))
    
    # Write output line by line
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(output_data)} entries")

def main():
    files_path = "/app/data/COLIEE/task1_test_files_2024/task1_test_files_2024"
    labels_file = "/app/data/COLIEE/task1_test_labels_2024.json"
    output_file = "/app/data/COLIEE/test_paragraphs_processed_data.json"

    process_files(files_path, labels_file, output_file)

if __name__ == "__main__":
    main()