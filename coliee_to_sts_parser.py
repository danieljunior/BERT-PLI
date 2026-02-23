import os
import json
from pathlib import Path
import random
from tqdm import tqdm
import ray
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from provenance.retrospective_service import RetrospectiveService
from provenance.prospective_service import ProspectiveService

random.seed(42)
nltk.download("punkt")
LANGUAGE = "english"

def read_text_file(file_path):
    """Read text file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

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

def process_files(files_path, labels_file, output_file_vanilla, output_file_sumy):
    # Read labels file
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Get all files in directory
    files = set(Path(files_path).glob('*.txt'))
    files = {f.name for f in files}
    
    # Generate positive and negative pairs
    positive_pairs = [(q, p) for q in labels for p in labels[q]]
    negative_pairs = generate_negative_pairs(labels, files)
    
    # Initialize Ray
    ray.init(num_cpus=6, ignore_reinit_error=True)

    @ray.remote
    def process_pair(q_file, p_file, files_path, positive_pairs_set):
        q_parser = PlaintextParser.from_file(f"{files_path}/{q_file}", Tokenizer(LANGUAGE))
        p_parser = PlaintextParser.from_file(f"{files_path}/{p_file}", Tokenizer(LANGUAGE))
        q_sentences = [str(sent) for sent in q_parser.document.sentences]
        p_sentences = [str(sent) for sent in p_parser.document.sentences]

        entry = {
            "guid": f"{q_file.split('.')[0]}_{p_file.split('.')[0]}",
            "q_paras": q_sentences,
            "c_paras": p_sentences,
            "label": 1 if (q_file, p_file) in positive_pairs_set else 0
        }
        return entry
    
    @ray.remote
    def sumy_process_pair(q_file, p_file, files_path, positive_pairs_set, percentual=0.5):
        stemmer = Stemmer(LANGUAGE)
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)

        q_parser = PlaintextParser.from_file(f"{files_path}/{q_file}", Tokenizer(LANGUAGE))
        q_total = max(1, int(len(q_parser.document.sentences) * percentual))
        q_summary = summarizer(q_parser.document, q_total)
        q_sentences = [str(sent) for sent in q_summary]
        
        p_parser = PlaintextParser.from_file(f"{files_path}/{p_file}", Tokenizer(LANGUAGE))
        p_total = max(1, int(len(p_parser.document.sentences) * percentual))
        p_summary = summarizer(p_parser.document, p_total)
        p_sentences = [str(sent) for sent in p_summary]

        entry = {
            "guid": f"{q_file.split('.')[0]}_{p_file.split('.')[0]}",
            "q_paras": q_sentences,
            "c_paras": p_sentences,
            "label": 1 if (q_file, p_file) in positive_pairs_set else 0
        }
        return entry

    positive_pairs_set = set(positive_pairs)
    all_pairs = positive_pairs + negative_pairs

    # Process all pairs - generate both vanilla and sumy versions simultaneously
    batch_size = 10
    vanilla_data = []
    sumy_data = []
    
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Batching Ray tasks"):
        batch = all_pairs[i:i + batch_size]
        
        # Dispatch both vanilla and sumy tasks for each pair
        vanilla_futures = []
        sumy_futures = []
        
        for q_file, p_file in tqdm(batch, desc="Dispatching Ray tasks", leave=False):
            vanilla_futures.append(process_pair.remote(q_file, p_file, files_path, positive_pairs_set))
            sumy_futures.append(sumy_process_pair.remote(q_file, p_file, files_path, positive_pairs_set))
        
        # Collect results in order
        vanilla_data.extend(ray.get(vanilla_futures))
        sumy_data.extend(ray.get(sumy_futures))
    
    # Write vanilla output
    with open(output_file_vanilla, 'w', encoding='utf-8') as f:
        for entry in vanilla_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Write sumy output
    with open(output_file_sumy, 'w', encoding='utf-8') as f:
        for entry in sumy_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(vanilla_data)} vanilla entries")
    print(f"Processed {len(sumy_data)} sumy entries")
    print(f"Vanilla output: {output_file_vanilla}")
    print(f"Sumy output: {output_file_sumy}")

def main():
    dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
    provenance = RetrospectiveService(dataflow_tag)

    files_path = "/app/data/COLIEE/task1_train_files_2024/"
    labels_file = "/app/data/COLIEE/task1_train_labels_2024.json"
    output_file_vanilla = "/app/data/COLIEE/train_vanilla_sentences.json"
    output_file_sumy = "/app/data/COLIEE/train_summarized_sentences.json"
    input_data = {"coliee_dataset": [[files_path, labels_file, "train"]]}
    with provenance.get_retrospective_data(ProspectiveService.TF_PARSE_COLIEE_DATASET, input_data) as result:
        process_files(files_path, labels_file, output_file_vanilla, output_file_sumy)
        result['coliee_parsed_dataset'] = [[output_file_vanilla, output_file_sumy, "train"]]

if __name__ == "__main__":
    main()