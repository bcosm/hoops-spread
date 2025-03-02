# Removes duplicate school names that map to the same code keeping the longest name
import json
import sys
import pathlib

# Processes school code mapping to keep only longest name per code
def dedupe(path: pathlib.Path) -> None:
    data = json.loads(path.read_text())
    
    longest = {}
    for k in sorted(data, key=len, reverse=True):
        code = data[k]
        if code not in longest:
            longest[code] = k

    cleaned = {name: code for code, name in longest.items()}
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        dedupe(pathlib.Path(sys.argv[1]))
