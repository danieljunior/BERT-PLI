import json
from tqdm import tqdm

original_labels_file = 'data/task1_train_labels_2024.json'
train_examples_file = 'data/train_full.json'
train_labels_file = 'data/train_labels.json'
valid_labels_file = 'data/valid_labels.json'

train_guids = set()
with open(train_examples_file, 'r') as f_train:
    for line in tqdm(f_train, desc="Reading train examples"):
        if line.strip():
            entry = json.loads(line)
            train_guids.add(entry['guid'])

train_labels = {}
valid_labels = {}

with open(original_labels_file, 'r') as f_original:
    entry = json.load(f_original)
    for guid in tqdm(train_guids):
        q = guid.split("_")[0]
        
        if f"{q}.txt" in train_labels:
            continue

        c = entry[q + ".txt"]
        for cf in c:
            cf = cf.split(".txt")[0]
            if f"{q}_{cf}" in train_guids:
                if  f"{q}.txt" in train_labels:
                    train_labels[f"{q}.txt"].append(cf+".txt")
                else:
                    train_labels[f"{q}.txt"] = [cf+".txt"]
            else:
                if  f"{q}.txt" in valid_labels:
                    valid_labels[f"{q}.txt"].append(cf+".txt")
                else:
                    valid_labels[f"{q}.txt"] = [cf+".txt"]                        
  
with open(train_labels_file, "w") as f:
    json.dump(train_labels, f, indent=4)
with open(valid_labels_file, "w") as f:
    json.dump(valid_labels, f, indent=4)
print("Splitting completed.")