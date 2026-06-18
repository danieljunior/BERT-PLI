import json
import argparse


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
from tqdm import tqdm

nltk.download("punkt_tab")

LANGUAGE = "english"

def merge_train_files(train_file, train_sumy_file, train_full_file, valid_full_file):
    """
    Split train.json based on whether GUIDs appear in train-sumy.json.
    
    Args:
        train_file: Path to train.json
        train_sumy_file: Path to train-sumy.json
        train_full_file: Path to output train-full.json (matching GUIDs)
        valid_full_file: Path to output valid-full.json (non-matching GUIDs)
    """
    # Read all GUIDs from train-sumy.json
    sumy_guids = set()
    with open(train_sumy_file, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sumy_guids.add(item['guid'])
    
    print(f"Found {len(sumy_guids)} unique GUIDs in {train_sumy_file}")
    
    # Process train.json and split based on GUID presence
    train_full_count = 0
    valid_full_count = 0
    
    with open(train_file, 'r') as f_in, \
         open(train_full_file, 'w') as f_train, \
         open(valid_full_file, 'w') as f_valid:
        
        for line in f_in:
            if line.strip():
                item = json.loads(line)
                guid = item['guid']
                
                if guid in sumy_guids:
                    f_train.write(line)
                    train_full_count += 1
                else:
                    f_valid.write(line)
                    valid_full_count += 1
    
    print(f"\nResults:")
    print(f"{train_full_file}: {train_full_count} samples (GUIDs in train-sumy.json)")
    print(f"{valid_full_file}: {valid_full_count} samples (GUIDs not in train-sumy.json)")


def parse_to_sumy_sentences(sumy_file, original_path, output_file):
    with open(sumy_file, 'r') as f_input, open(output_file, 'w') as f_output:
        for line in tqdm(f_input, desc="Parsing sumy file"):
            if line.strip():
                item = json.loads(line)
                q_path, p_path = item['guid'].split('_')
                q_file = f"{original_path}/{q_path}.txt"
                p_file = f"{original_path}/{p_path}.txt"
                q_parser = PlaintextParser.from_file(q_file, Tokenizer(LANGUAGE))
                p_parser = PlaintextParser.from_file(p_file, Tokenizer(LANGUAGE))
                q_sentences = [str(sent) for sent in q_parser.document.sentences]
                p_sentences = [str(sent) for sent in p_parser.document.sentences]
            
                entry = {
                    "guid": item['guid'],
                    "q_paras": q_sentences,
                    "c_paras": p_sentences,
                    "label": item['label'],
                }
                f_output.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split train.json based on GUID presence in train-sumy.json"
    )
    # parser.add_argument("--train-file", default="data/train_paragraphs_processed_data.json", 
    #                     help="Path to train.json (default: data/train_paragraphs_processed_data.json)")
    # parser.add_argument("--train-sumy-file", default="data/train_sumy.json",
    #                     help="Path to train-sumy.json (default: data/train_sumy.json)")
    # parser.add_argument("--train-full-file", default="data/train_full.json",
    #                     help="Path to output train_full.json (default: data/train_full.json)")
    # parser.add_argument("--valid-full-file", default="data/valid_full.json",
    #                     help="Path to output valid_full.json (default: data/valid_full.json)")

    # args = parser.parse_args()
    
    # merge_train_files(
    #     args.train_file,
    #     args.train_sumy_file,
    #     args.train_full_file,
    #     args.valid_full_file
    # )
    parser.add_argument("--sumy-file", default="data/train_sumy.json", 
                        help="Path to train.json (default: data/train_sumy.json)")
    parser.add_argument("--original-path", default="data/task1_train_files_2024/task1_train_files_2024/",
                        help="Path to train-sumy.json (default: data/task1_train_files_2024/task1_train_files_2024/)")
    parser.add_argument("--output-file", default="data/train_full.json",
                        help="Path to output train_full.json (default: data/train_full.json)")
    
    args = parser.parse_args()
    parse_to_sumy_sentences(
        args.sumy_file,
        args.original_path,
        args.output_file
    )