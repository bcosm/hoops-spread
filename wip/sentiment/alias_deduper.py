# Removes duplicate aliases that map to multiple schools using LLM disambiguation
import json
import os
import requests
from collections import defaultdict
from typing import Dict, List, Optional

INPUT_FILE = "./data/processed_with_school_codes/school_aliases.jsonl"
API_URL = "http://127.0.0.1:1234/v1"
MODEL = "deepseek-r1-distill-llama-8b"

# Queries LLM to determine which school an ambiguous alias should map to
def query_model(alias: str, schools: List[str]) -> Optional[str]:
    try:
        payload = {
            "model": MODEL,
            "stream": False,
            "max_tokens": 1000,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Reply only with the exact school name from the list."},
                {"role": "user", "content": f'Alias: "{alias}"\nSchools: {", ".join(schools)}\nWhich school?'}
            ]
        }
        
        response = requests.post(f"{API_URL}/chat/completions", json=payload, timeout=30)
        response.raise_for_status()
        
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        for school in schools:
            if school.lower() in content.lower():
                return school
        
        return None
    except Exception:
        return None

# Loads school alias mappings from JSONL file
def load_aliases(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    
    data = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                data[obj["school"]] = obj["aliases"]
    except Exception:
        return {}
    return data

# Saves updated alias mappings back to JSONL file
def save_aliases(path: str, mapping: Dict[str, List[str]]) -> None:
    try:
        with open(path, 'w') as f:
            for school, aliases in sorted(mapping.items()):
                f.write(json.dumps({"school": school, "aliases": sorted(aliases)}) + "\n")
    except Exception:
        pass

# Identifies duplicate aliases and resolves conflicts using LLM
def main():
    data = load_aliases(INPUT_FILE)
    if not data:
        return

    rev_map = defaultdict(list)
    for school, alias_list in data.items():
        for alias in alias_list:
            rev_map[alias].append(school)

    duplicates = {a: s for a, s in rev_map.items() if len(s) > 1}
    
    if not duplicates:
        return

    for alias, schools in duplicates.items():
        winner = query_model(alias, schools)
        if winner:
            for s in schools:
                if s != winner and alias in data[s]:
                    data[s].remove(alias)

    save_aliases(INPUT_FILE, data)

if __name__ == "__main__":
    main()
