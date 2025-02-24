# Scrapes Sports Reference to find coaching start dates and pre-dataset statistics
import json
import time
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import io

INPUT_CSV = Path("data/features/ncaab_cumulative_features_v1.csv")
OUTPUT_DIR = Path("data/features")
COACH_DATA_CSV = OUTPUT_DIR / "coach_sr_true_starts_and_stats.csv"

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "deepseek-r1-distill-llama-8b"
MAX_RETRIES = 2
RETRY_TEMPS = [0.3, 0.5]
RETRY_DELAY = 10
TIMEOUT = 30
REQUEST_DELAY = 2
MAX_CHUNK_LENGTH = 3500

total_tokens = 0


                          
# Constructs Sports Reference URL for coach profile based on name and attempt number
def build_coach_url(coach_name, attempt=1):
    if not coach_name or pd.isna(coach_name):
        return None
    
    name = str(coach_name).lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-+", "-", name).strip('-')
    
    if not name:
        return None
        
    slug = f"{name}-{attempt}"
    return f"https://www.sports-reference.com/cbb/coaches/{slug}.html"

# Uses LLM to extract specific team coaching seasons from Sports Reference table data
def extract_seasons(coach_name, target_team, table_text):
    global total_tokens

    prompt = f"""Analyze coaching record for {coach_name}.
Identify seasons where {coach_name} coached "{target_team}".
School names may vary (e.g., "St. John's" vs "St John's (NY)", "California" vs "Cal").

Table Data:
---
{table_text}
---

Return JSON with "seasons_at_target_team" key containing list of seasons (e.g., ["2005-06", "2006-07"]).
If no matches, return empty list.
LSU and Louisiana State are the same team.
"""
    
    temps = [0.1] + RETRY_TEMPS
    
    for attempt, temp in enumerate(temps):
        if attempt > 0:
            time.sleep(RETRY_DELAY / 2)

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Return JSON responses only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": { 
                "type": "json_schema", 
                "json_schema": {      
                    "name": "coach_season_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "seasons_at_target_team": {
                                "type": "array",
                                "items": {"type": "string"} 
                            }
                        },
                        "required": ["seasons_at_target_team"]
                    }
                }
            },
            "temperature": temp,
            "max_tokens": 300,
            "stream": False
        }
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = requests.post(API_URL, headers={"Content-Type": "application/json"}, 
                                       json=payload, timeout=120)
                
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        if "error" in error_data and "model has crashed" in error_data["error"]:
                            return "MODEL_CRASHED"
                    except:
                        pass

                response.raise_for_status()
                data = response.json()
                
                usage = data.get("usage", {})
                if "total_tokens" in usage:
                    total_tokens += usage["total_tokens"]

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    retries += 1
                    time.sleep(RETRY_DELAY)
                    continue
                
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "seasons_at_target_team" in parsed:
                    seasons = parsed["seasons_at_target_team"]
                    if isinstance(seasons, list):
                        if seasons:
                            return seasons
                        elif attempt < len(temps) - 1:
                            break
                        else:
                            return []
                
                if attempt == len(temps) - 1:
                    return "FORMAT_ERROR"
                else:
                    break

            except requests.exceptions.HTTPError:
                if "model has crashed" in response.text:
                    return "MODEL_CRASHED"
                retries += 1
                time.sleep(RETRY_DELAY * (retries + 1))
            except Exception:
                retries += 1
                time.sleep(RETRY_DELAY * (retries + 1))
        
        if retries == MAX_RETRIES and attempt == len(temps) - 1:
            return "CALL_FAILED"
            
    return "CALL_FAILED"                                             


# Processes all coach-team pairs to find start seasons and pre-dataset statistics
def main():
    global total_tokens
    total_tokens = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        return
    
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        df['date_obj'] = pd.to_datetime(df['date'])
    except Exception:
        return

    df.sort_values(by=['date_obj'], inplace=True)
    earliest_season = df['season'].min()
    if not earliest_season or not isinstance(earliest_season, str):
        return
    
    try:
        earliest_end_year = int(earliest_season.split('-')[0]) + 1
    except:
        return

    first_coach_team_game_df = df.drop_duplicates(subset=['visitor_bart_team_coach', 'visitor_team'], keep='first')
    first_coach_team_game_df = pd.concat([
        first_coach_team_game_df[['visitor_bart_team_coach', 'visitor_team', 'season']].rename(
            columns={'visitor_bart_team_coach':'coach_name', 'visitor_team':'team_name'}),
        df.drop_duplicates(subset=['home_bart_team_coach', 'home_team'], keep='first')[
            ['home_bart_team_coach', 'home_team', 'season']].rename(
            columns={'home_bart_team_coach':'coach_name', 'home_team':'team_name'})
    ]).drop_duplicates(subset=['coach_name', 'team_name'], keep='first')

    coach_team_pairs = set()
    for _, row in first_coach_team_game_df.iterrows():
        coach = str(row['coach_name'])
        team = str(row['team_name'])
        first_season = str(row['season'])
        if (pd.notna(coach) and coach != "nan" and pd.notna(team) and 
            team != "nan" and first_season == earliest_season):
            coach_team_pairs.add((coach, team))
            
    if not coach_team_pairs:
        return

    researched_data = []
    if COACH_DATA_CSV.exists():
        try:
            researched_df = pd.read_csv(COACH_DATA_CSV, dtype=str)
            researched_data = researched_df.to_dict('records')
        except Exception:
            pass
    
    existing_pairs = set((item['coach_name'], item['team_name']) for item in researched_data)
    pairs_to_process = sorted([pair for pair in list(coach_team_pairs) if pair not in existing_pairs])
    
    if not pairs_to_process:
        return

    session = requests.Session()
    retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    for coach_name, target_team in tqdm(pairs_to_process, desc="Processing coaches"):
        if pd.isna(coach_name) or pd.isna(target_team) or coach_name == "nan" or target_team == "nan": 
            continue

        coach_tables_df = None
        
        for attempt in range(1, 7):
            url = build_coach_url(coach_name, attempt)
            if not url:
                break
            
            try:
                response = session.get(url, timeout=TIMEOUT)
                time.sleep(REQUEST_DELAY)
                if response.status_code == 200:
                    tables = pd.read_html(io.StringIO(response.text), attrs={'id': 'coaching-record'})
                    if tables:
                        coach_tables_df = tables[0]
                        break
            except Exception:
                pass

        if coach_tables_df is None or coach_tables_df.empty:
            researched_data.append({
                'coach_name': coach_name, 'team_name': target_team, 'sr_url_used': url if url else "N/A",
                'true_start_season_sr': 'NOT_FOUND', 
                'games_before_dataset_at_team': 0, 'wins_before_dataset_at_team': 0, 
                'losses_before_dataset_at_team': 0, 'llm_notes': 'SR page/table fetch failed'
            })
            continue

        try:
            if isinstance(coach_tables_df.columns, pd.MultiIndex):
                coach_tables_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                         for col in coach_tables_df.columns.values]
            
            season_col = next((col for col in coach_tables_df.columns if 'season' in col.lower()), None)
            school_col = next((col for col in coach_tables_df.columns if 'school' in col.lower()), None)
            
            if not season_col or not school_col:
                researched_data.append({'coach_name': coach_name, 'team_name': target_team, 
                                      'sr_url_used': url, 'true_start_season_sr': 'PARSE_ERROR', 
                                      'games_before_dataset_at_team':0, 'wins_before_dataset_at_team':0, 
                                      'losses_before_dataset_at_team':0, 'llm_notes': 'Column missing'})
                continue

            filtered_df = coach_tables_df[~coach_tables_df[season_col].astype(str).str.contains(
                "Career|Overall", case=False, na=False)]
            table_text = filtered_df[[season_col, school_col]].to_string(index=False)
            
            if len(table_text) > MAX_CHUNK_LENGTH * 3:
                table_text = table_text[:MAX_CHUNK_LENGTH*3] + "\n...[TRUNCATED]"

        except Exception:
            researched_data.append({'coach_name': coach_name, 'team_name': target_team, 
                                  'sr_url_used': url, 'true_start_season_sr': 'PREP_ERROR', 
                                  'games_before_dataset_at_team':0, 'wins_before_dataset_at_team':0, 
                                  'losses_before_dataset_at_team':0, 'llm_notes': 'Table prep error'})
            continue
            
        seasons = extract_seasons(coach_name, target_team, table_text)

        true_start_season = None
        games_before = 0
        wins_before = 0
        losses_before = 0
        notes = ""

        if isinstance(seasons, list) and seasons:
            earliest_end_year_for_team = float('inf')
            
            for season_str in seasons:
                try:
                    if isinstance(season_str, str) and '-' in season_str:
                        start_yr_str = season_str.split('-')[0]
                        if len(start_yr_str) == 4 and start_yr_str.isdigit():
                            current_end_year = int(start_yr_str) + 1
                            if current_end_year < earliest_end_year_for_team:
                                earliest_end_year_for_team = current_end_year
                                true_start_season = season_str
                except Exception:
                    pass
            
            if true_start_season:
                for _, sr_row in filtered_df.iterrows():
                    try:
                        sr_season = str(sr_row[season_col])
                        sr_school = str(sr_row[school_col])
                        
                        if (target_team.lower() in sr_school.lower() or 
                            sr_school.lower() in target_team.lower()):
                            if '-' in sr_season:
                                sr_start_yr = sr_season.split('-')[0]
                                if len(sr_start_yr) == 4 and sr_start_yr.isdigit():
                                    sr_end_year = int(sr_start_yr) + 1
                                    if sr_end_year < earliest_end_year:
                                        games_col = next((col for col in coach_tables_df.columns 
                                                        if col.lower() in ['g', 'games']), None)
                                        wins_col = next((col for col in coach_tables_df.columns 
                                                       if col.lower() in ['w', 'wins']), None)
                                        losses_col = next((col for col in coach_tables_df.columns 
                                                         if col.lower() in ['l', 'losses']), None)

                                        if games_col and wins_col and losses_col:
                                            games_val = pd.to_numeric(sr_row[games_col], errors='coerce')
                                            wins_val = pd.to_numeric(sr_row[wins_col], errors='coerce')
                                            losses_val = pd.to_numeric(sr_row[losses_col], errors='coerce')

                                            games_before += int(games_val) if pd.notna(games_val) else 0
                                            wins_before += int(wins_val) if pd.notna(wins_val) else 0
                                            losses_before += int(losses_val) if pd.notna(losses_val) else 0
                    except Exception:
                        pass
                notes = "Extracted from SR"
            else:
                notes = "Seasons found but earliest unclear"
        elif isinstance(seasons, list) and not seasons:
            notes = "No matching seasons found"
        else:
            notes = f"Processing error: {seasons}"

        researched_data.append({
            'coach_name': coach_name, 
            'team_name': target_team, 
            'sr_url_used': url if url else "N/A",
            'true_start_season_sr': true_start_season if true_start_season else "NOT_FOUND",
            'games_before_dataset_at_team': games_before,
            'wins_before_dataset_at_team': wins_before,
            'losses_before_dataset_at_team': losses_before,
            'llm_notes': notes
        })

        try:
            pd.DataFrame(researched_data).drop_duplicates(
                subset=['coach_name', 'team_name'], keep='last').to_csv(COACH_DATA_CSV, index=False)
        except Exception:
            pass
                
    try:
        if researched_data:
            final_df = pd.DataFrame(researched_data).drop_duplicates(
                subset=['coach_name', 'team_name'], keep='last')
            final_df.to_csv(COACH_DATA_CSV, index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
