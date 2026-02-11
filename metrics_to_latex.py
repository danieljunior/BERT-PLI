import json
import pandas as pd
from pathlib import Path
import sys


def load_metrics_to_dataframe(json_files: list) -> pd.DataFrame:
    """
    Load multiple JSON metrics files into a DataFrame.
    
    Args:
        json_files: List of JSON file paths
        
    Returns:
        DataFrame with JSON keys as columns and filenames as index
    """
    data = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            metrics = json.load(f)
            split_path = json_file.split('/')
            model = split_path[-2]+ split_path[-1].split('_')[0]
            data[model] = metrics
    
    df = pd.DataFrame.from_dict(data, orient='index')
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_files = sys.argv[1:]
        df = load_metrics_to_dataframe(json_files)
        print(df)
        print(df.info())
        print(df.to_latex(index=True))
    else:
        print("Please provide at least one JSON file as an argument.")
   