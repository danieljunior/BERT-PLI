import json
import sys

def verify_guids(file_a, file_b):
    """Verify if all 'guid' values from file A are present in file B."""
    
    # Load guids from file A
    guids_a = set()
    try:
        with open(file_a, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if 'guid' in obj:
                    guids_a.add(obj['guid'])
    except Exception as e:
        print(f"Error reading file A: {e}")
        return False
    
    # Load guids from file B
    guids_b = set()
    try:
        with open(file_b, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if 'guid' in obj:
                    guids_b.add(obj['guid'])
    except Exception as e:
        print(f"Error reading file B: {e}")
        return False
    
    # Check if all guids from A are in B
    missing = guids_a - guids_b
    
    if missing:
        print(f"Found {len(missing)} guids from A not in B:")
        for guid in sorted(missing):
            print(f"  - {guid}")
        print(f"✗ {len(missing)} guids from A are missing in B")
        return False
    else:
        print(f"✓ All {len(guids_a)} guids from A are present in B")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_data.py <file_a> <file_b>")
        sys.exit(1)
    
    verify_guids(sys.argv[1], sys.argv[2])