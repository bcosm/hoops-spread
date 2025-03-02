# Generates school aliases using LLM for college basketball sentiment analysis
import json
import os
import pathlib
import re
import requests
from typing import Dict, List, Optional

# Safely parses JSON response from LLM extracting alias list
def safe_json_parse(txt: str) -> Optional[List[str]]:
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*]", txt)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, list) else None
            except json.JSONDecodeError:
                pass
        return None

# Processes college scorecards to generate comprehensive alias mappings using LLM
def main():
    api_key = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key")
    input_path = pathlib.Path("./data/processed_with_school_codes/scorecard_master.csv")
    output_path = pathlib.Path("./data/processed_with_school_codes/school_aliases.jsonl")
    
    if not input_path.exists():
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        schools = json.loads(input_path.read_text())
    except Exception:
        return

    processed_schools = set()
    if output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "school" in data:
                            processed_schools.add(data["school"])
                    except json.JSONDecodeError:
                        continue

    schools_to_process = [s for s in schools if s not in processed_schools]
    
    if not schools_to_process:
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with output_path.open("a") as fout:
        for school in schools_to_process:
            payload = {
                "model": "deepseek/deepseek-r1:free",
                "max_tokens": 800,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "Respond ONLY with a valid JSON array (max 20 strings, lowercase)."},
                    {"role": "user", "content": f"return a json array (max 20 items) of every possible alias for {school.replace('_', ' ')} men's ncaa basketball, or the school in general, that may be used in online discussions (reddit, discord, twitter, etc). all lowercase."}
                ],
            }

            aliases = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    if resp.status_code == 429:
                        continue
                    resp.raise_for_status()

                    raw = resp.json()["choices"][0]["message"]["content"]
                    aliases = safe_json_parse(raw)
                    if aliases is not None:
                        break

                except requests.RequestException:
                    continue

            if aliases is None:
                aliases = []

            fout.write(json.dumps({"school": school, "aliases": aliases}) + "\n")
            fout.flush()

if __name__ == "__main__":
    main()