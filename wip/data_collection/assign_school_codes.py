# Assign federal OPEID codes to school names using College Scorecard API and LLM matching
import os
import json
import time
import re
import math
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from rapidfuzz import process, fuzz
from tqdm import tqdm
import traceback
import subprocess

API_KEY = "your_college_scorecard_api_key"

LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "deepseek-r1-distill-llama-8b"
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 10
LLM_CANDIDATES_COUNT = 30
GPU_TEMP_CUTOFF = 85

SBR_ELO_DATA_PATH = Path("data/processed/sbr_ncaab_with_elo_ratings.csv")
RAW_BART_TORVIK_DATA_CSV = Path("data/raw/bart_torvik_full_raw_data.csv")

OUTPUT_DIR = Path("data/processed_with_school_codes")
MASTER_LIST_CACHE = OUTPUT_DIR / "scorecard_master.csv"
LLM_SCHOOL_NAME_TO_CODE_CACHE_FILE = OUTPUT_DIR / "llm_school_name_to_code.json"
CODE_TO_SBR_ORIGIN_NAME_CACHE = OUTPUT_DIR / "llm_code_to_sbr_origin_name.json"
LLM_NO_MATCH_LOG_FILE = OUTPUT_DIR / "llm_no_match_log.json"

SBR_WITH_CODES_CSV = OUTPUT_DIR / "sbr_elo_with_llm_codes.csv"
BART_WITH_CODES_CSV = OUTPUT_DIR / "bart_torvik_with_llm_codes.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_gpu_temperature() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        temps = [int(t) for t in output.splitlines() if t.strip().isdigit()]
        return max(temps) if temps else 0
    except Exception:
        return 0

# Load or fetch the master list of schools with OPEID codes from College Scorecard API
def load_master_list() -> pd.DataFrame:
    if MASTER_LIST_CACHE.exists():
        try:
            df_cached = pd.read_csv(MASTER_LIST_CACHE, dtype={'opeid': str})
            if {"name", "opeid"}.issubset(df_cached.columns):
                df_cached['opeid'] = df_cached['opeid'].astype(str).replace('nan', pd.NA).str.strip()
                df_cached['name'] = df_cached['name'].astype(str).str.strip()
                return df_cached
        except Exception:
            pass

    session = requests.Session()
    retry_strategy = Retry(total=5, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    base = "https://api.data.gov/ed/collegescorecard/v1/schools"
    per_page = 100
    meta_params = {"api_key": API_KEY, "fields": "ope6_id", "per_page": 1}
    
    try:
        resp_meta = session.get(base, params=meta_params, timeout=30)
        resp_meta.raise_for_status()
        total_records = resp_meta.json()["metadata"]["total"]
        total_pages = math.ceil(total_records / per_page)
    except Exception as e:
        raise RuntimeError(f"Could not fetch metadata: {e}") from e

    all_results = []
    fetch_params = {"api_key": API_KEY, "fields": "school.name,ope6_id", "per_page": per_page}
    
    for page in tqdm(range(total_pages), desc="Fetching pages", unit="page"):
        fetch_params["page"] = page
        try:
            r = session.get(base, params=fetch_params, timeout=30)
            r.raise_for_status()
            results = r.json().get("results", [])
            all_results.extend(results)
            time.sleep(0.1)
        except Exception:
            continue
            
    df_raw = pd.DataFrame(all_results)
    rename_map = {}
    if "school.name" in df_raw.columns:
        rename_map["school.name"] = "name"
    if "ope6_id" in df_raw.columns:
        rename_map["ope6_id"] = "opeid"
    if rename_map:
        df_raw.rename(columns=rename_map, inplace=True)

    if not {"name", "opeid"}.issubset(df_raw.columns) or df_raw.empty:
        raise RuntimeError(f"Fetched master list missing required columns or is empty. Found: {df_raw.columns.tolist()}")

    df_master = df_raw[["name", "opeid"]].copy()
    df_master['name'] = df_master['name'].astype(str).str.strip()
    df_master['opeid'] = df_master['opeid'].astype(str).replace('nan', pd.NA).str.strip()
    df_master.dropna(subset=['name', 'opeid'], inplace=True)
    df_master.to_csv(MASTER_LIST_CACHE, index=False)
    return df_master

# Normalize school names for better fuzzy matching by standardizing abbreviations and formatting
def normalize(name: str) -> str:
    s = str(name).lower()
    replacements = [
        (" u ", " university "), (" st ", " state "), (" coll ", " college "),
        (" inst ", " institute "), (" tech ", " technology "), (" & ", " and "),
        (" cal ", " california "), (" fla ", " florida "), (" ny ", " new york "),
        (" nc ", " north carolina "), (" sc ", " south carolina "),
        ("va ", "virginia "), (" mich ", " michigan "), ("wis ", "wisconsin "),
        ("tex ", "texas "), ("ga ", "georgia "), ("pa ", "pennsylvania "),
        ("la ", "louisiana "), ("'", "")
    ]
    for k, v in replacements:
        s = s.replace(k, v)
    s = re.sub(r" at ", " ", s)
    s = re.sub(r" of ", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# Call LLM API to match school name to OPEID using fuzzy search candidates
def call_llm_for_opeid(school_name_to_match: str, candidate_schools: List[Dict[str, str]]) -> Optional[str]:
    while get_gpu_temperature() > GPU_TEMP_CUTOFF:
        time.sleep(15)

    candidate_list_str = json.dumps(candidate_schools, indent=2)
    prompt = f"""You are an expert in US higher education institutions and their official OPEID codes.
Your task is to match the provided "School Name to Match" to the most appropriate school from the "Candidate List".

School Name to Match: "{school_name_to_match}"

Candidate List (Official Name and OPEID):
{candidate_list_str}

Consider variations in naming, abbreviations, and common terms (e.g., "State" for "State University", "A&M" for "Agricultural and Mechanical").
Based on all available information, which OPEID from the Candidate List is the correct and most confident match for "{school_name_to_match}"?

Return your answer as a JSON object with a single key "opeid".
If you are highly confident in a match from the Candidate List, provide its 6-digit OPEID string. Example: {{"opeid": "001234"}}
If none of the candidates are a good match or you are not confident, the value for "opeid" should be null. Example: {{"opeid": null}}
Only choose an OPEID that is present in the provided Candidate List. Do not invent new OPEIDs or guess one not listed.
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that meticulously follows instructions and returns responses in the specified JSON format."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "opeid_match_response",
                "description": "Provides the matched OPEID (string) or null if no confident match.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "opeid": {
                            "type": ["string", "null"]
                        }
                    },
                    "required": ["opeid"]
                }
            }
        },
        "temperature": 0.2,
        "max_tokens": 150,
        "stream": False
    }

    retries = 0
    response_text = ""
    content_str = ""

    while retries < LLM_MAX_RETRIES:
        try:
            response = requests.post(LLM_API_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=120)
            response_text = response.text
            response.raise_for_status()
            data = response.json()
            
            content_str = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content_str:
                retries += 1
                time.sleep(LLM_RETRY_DELAY)
                continue

            parsed_content = json.loads(content_str)
            if "opeid" in parsed_content:
                opeid_result = parsed_content["opeid"]
                if opeid_result is None:
                    return None
                    
                if isinstance(opeid_result, str) and re.match(r"^[0-9A-Za-z]{1,8}$", opeid_result):
                    candidate_opeids = {c['opeid'] for c in candidate_schools}
                    if opeid_result in candidate_opeids:
                        return opeid_result
                    else:
                        return None
                elif opeid_result is not None:
                     return None
            else:
                return None

        except requests.exceptions.RequestException:
            pass
        except json.JSONDecodeError:
            pass
        except Exception:
            pass

        retries += 1
        time.sleep(LLM_RETRY_DELAY * (retries + 1))
    
    return None

# Load JSON cache from file with error handling
def load_json_cache(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}

# Save JSON data to cache file with proper encoding
def save_json_cache(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# Process dataset by extracting unique school names and matching them to OPEID codes using LLM
def process_dataset_with_llm(
    df: pd.DataFrame,
    name_cols: List[str],
    llm_school_name_to_code: Dict[str, Optional[str]],
    code_to_origin: Dict[str, str],
    master_df: pd.DataFrame,
    dataset_name: str,
    is_sbr: bool = False
):
    unique_school_names = set()
    for col in name_cols:
        if col in df.columns:
            unique_school_names.update(df[col].dropna().astype(str).unique())
    
    names_for_llm = sorted([name for name in list(unique_school_names) if name not in llm_school_name_to_code])
    
    if not names_for_llm:
        return

    master_df_cleaned = master_df.dropna(subset=['name', 'opeid']).copy()
    if master_df_cleaned.empty:
        for school_name in names_for_llm:
            llm_school_name_to_code[school_name] = None
        return

    master_df_cleaned['normalized_name'] = master_df_cleaned['name'].apply(normalize)
    normalized_master_choices = master_df_cleaned['normalized_name'].tolist()
    
    if not normalized_master_choices:
        for school_name in names_for_llm:
            llm_school_name_to_code[school_name] = None
        return

    llm_no_match_log = load_json_cache(LLM_NO_MATCH_LOG_FILE)

    for i, school_name in enumerate(tqdm(names_for_llm, desc=f"LLM Matching [{dataset_name}]", unit="school")):
        normalized_input_name = normalize(school_name)
        
        fuzzy_candidates = process.extract(
            normalized_input_name, normalized_master_choices, 
            scorer=fuzz.WRatio, limit=LLM_CANDIDATES_COUNT, score_cutoff=10
        )
        
        llm_candidate_list = []
        if fuzzy_candidates:
            for norm_master_name, score, idx in fuzzy_candidates:
                original_master_record = master_df_cleaned.iloc[idx]
                llm_candidate_list.append({
                    "official_name": original_master_record['name'],
                    "opeid": original_master_record['opeid']
                })
        
        if not llm_candidate_list:
            llm_school_name_to_code[school_name] = None
            llm_no_match_log[school_name] = {
                "reason": "No fuzzy candidates generated", 
                "dataset": dataset_name, 
                "normalized_name": normalized_input_name
            }
            continue

        opeid_from_llm = call_llm_for_opeid(school_name, llm_candidate_list)
        llm_school_name_to_code[school_name] = opeid_from_llm

        if opeid_from_llm:
            if is_sbr and opeid_from_llm not in code_to_origin:
                code_to_origin[opeid_from_llm] = school_name
        else:
            llm_no_match_log[school_name] = {
                "reason": "LLM returned null or failed", 
                "dataset": dataset_name, 
                "candidates_provided_count": len(llm_candidate_list), 
                "normalized_name": normalized_input_name
            }

        if (i + 1) % 20 == 0:
            save_json_cache(llm_school_name_to_code, LLM_SCHOOL_NAME_TO_CODE_CACHE_FILE)
            if is_sbr:
                save_json_cache(code_to_origin, CODE_TO_SBR_ORIGIN_NAME_CACHE)
            save_json_cache(llm_no_match_log, LLM_NO_MATCH_LOG_FILE)

    save_json_cache(llm_school_name_to_code, LLM_SCHOOL_NAME_TO_CODE_CACHE_FILE)
    if is_sbr:
        save_json_cache(code_to_origin, CODE_TO_SBR_ORIGIN_NAME_CACHE)
    save_json_cache(llm_no_match_log, LLM_NO_MATCH_LOG_FILE)

# Transform dataframe by adding federal code columns based on school name mappings
def transform_df_with_codes(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    school_name_to_code_map: Dict[str, Optional[str]],
    code_to_origin_map: Dict[str, str]
) -> pd.DataFrame:
    df2 = df.copy()
    for orig_col, code_col in mapping.items():
        if orig_col not in df2.columns:
            continue
        df2[code_col] = df2[orig_col].astype(str).map(lambda x: school_name_to_code_map.get(x))
        origin_col = code_col.replace("_federal_code", "_original_sbr_school_name")
        df2[origin_col] = df2[code_col].map(lambda c: code_to_origin_map.get(c) if pd.notna(c) else None)
    return df2

# Main function to orchestrate the school code assignment process for both datasets
def main():
    master_df = load_master_list()
    if master_df.empty or not {"name", "opeid"}.issubset(master_df.columns):
        return

    llm_school_name_to_code = load_json_cache(LLM_SCHOOL_NAME_TO_CODE_CACHE_FILE)
    code_to_origin = load_json_cache(CODE_TO_SBR_ORIGIN_NAME_CACHE)

    try:
        sbr_df = pd.read_csv(SBR_ELO_DATA_PATH, dtype=str)
        process_dataset_with_llm(
            sbr_df, ["visitor_team", "home_team"], 
            llm_school_name_to_code, code_to_origin, 
            master_df, "SBR/Elo", is_sbr=True
        )
        sbr_transformed_df = transform_df_with_codes(
            sbr_df, 
            {"visitor_team": "visitor_federal_code", "home_team": "home_federal_code"}, 
            llm_school_name_to_code, code_to_origin
        )
        sbr_transformed_df.to_csv(SBR_WITH_CODES_CSV, index=False)
    except FileNotFoundError:
        pass
    except Exception:
        pass

    try:
        bart_df = pd.read_csv(RAW_BART_TORVIK_DATA_CSV, dtype=str, low_memory=False)
        process_dataset_with_llm(
            bart_df, ["bart_team", "bart_opponent"], 
            llm_school_name_to_code, code_to_origin,
            master_df, "Bart Torvik", is_sbr=False
        )
        bart_transformed_df = transform_df_with_codes(
            bart_df, 
            {"bart_team": "team_federal_code", "bart_opponent": "opponent_federal_code"}, 
            llm_school_name_to_code, code_to_origin
        )
        bart_transformed_df.to_csv(BART_WITH_CODES_CSV, index=False)
    except FileNotFoundError:
        pass
    except Exception:
        pass

if __name__ == "__main__":
    main()