import os
import json
from pathlib import Path

def read_text_file(file_path):
    """Read text file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def process_corpus(corpus_path, labels_file):
    # Read labels file
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    output_data = []
    
    # Iterate through all folders in corpus
    for folder in sorted(os.listdir(corpus_path)):
        folder_path = Path(corpus_path) / folder
        
        # Skip if not a directory
        if not folder_path.is_dir():
            continue
            
        # Read entailed fragment
        entailed_fragment = read_text_file(folder_path / 'entailed_fragment.txt')
        
        # Process each paragraph file
        paragraphs_path = folder_path / 'paragraphs'
        for para_file in sorted(os.listdir(paragraphs_path)):
            if not para_file.endswith('.txt'):
                continue
                
            para_id = para_file.split('.')[0]
            para_content = read_text_file(paragraphs_path / para_file)
            
            # Determine label
            label = 0
            if folder in labels and para_file in labels[folder]:
                label = 1
                
            # Create entry
            entry = {
                "guid": f"{folder}_{para_id}",
                "text_a": entailed_fragment,
                "text_b": para_content,
                "label": label
            }
            
            output_data.append(entry)
    
    return output_data

def main():
    corpus_path = "/app/data/COLIEE/task2_train_files_2024/task2_train_files_2024"
    labels_file = "/app/data/COLIEE/task2_train_labels_2024.json"
    output_file = "/app/data/COLIEE/train_entailment_processed_data.json"

    # Process corpus
    data = process_corpus(corpus_path, labels_file)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(data)} entries")

if __name__ == "__main__":
    main()