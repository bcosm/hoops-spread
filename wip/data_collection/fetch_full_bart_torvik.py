# Fetches comprehensive game statistics data from Bart Torvik website for college basketball analysis
import os
import pandas as pd
import requests 
import io 
import time 
import re 
import glob 

BASE_URL_PATTERN = "https://barttorvik.com/getgamestats.php?year={year}&csv=1"
SBR_ELO_DATA_PATH = os.path.join("data", "processed", "sbr_ncaab_with_elo_ratings.csv") 
OUTPUT_DIR_RAW = os.path.join("data", "raw")
BART_TORVIK_FULL_OUTPUT_CSV = os.path.join(OUTPUT_DIR_RAW, "bart_torvik_full_raw_data.csv") 

REQUEST_DELAY_SECONDS = 2.5 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/csv,application/csv,application/vnd.ms-excel,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

EXPECTED_BART_COLUMN_NAMES_FULL = [
    'bart_date_raw', 'bart_game_type', 'bart_team', 'bart_team_conf', 'bart_opponent', 
    'bart_venue', 'bart_result_string', 
    'bart_adj_o_ingame', 'bart_adj_d_ingame', 'bart_tempo_ingame', 
    'bart_off_eff_ingame', 'bart_off_efg_pct_ingame', 'bart_off_to_pct_ingame', 
    'bart_off_orb_pct_ingame', 'bart_off_ftr_ingame', 
    'bart_def_eff_ingame', 'bart_def_efg_pct_ingame', 'bart_def_to_pct_ingame', 
    'bart_def_drb_pct_ingame', 'bart_def_ftr_ingame', 
    'bart_opponent_conf_abb',       
    'bart_opponent_conf_order',     
    'bart_season_col',              
    'bart_possessions_game',        
    'bart_game_id_str',             
    'bart_team_coach',              
    'bart_opponent_coach',          
    'bart_proj_score_diff',         
    'bart_pred_barthag_wp',         
    'bart_box_score_array_str',     
    'bart_row_id_or_flag'           
]
NUM_EXPECTED_COLS_FULL = len(EXPECTED_BART_COLUMN_NAMES_FULL)

# Extracts season year range from existing SBR data to determine which seasons to fetch
def get_season_range_from_sbr_data_content(sbr_elo_filepath):
    if not os.path.exists(sbr_elo_filepath):
        return None, None
    
    try:
        try:
            df = pd.read_csv(sbr_elo_filepath, usecols=['season', 'season_end_year'])
        except ValueError:
            df = pd.read_csv(sbr_elo_filepath)
        
        oldest_season_end_year = None
        newest_season_end_year = None

        if 'season_end_year' in df.columns:
            df['season_end_year'] = pd.to_numeric(df['season_end_year'], errors='coerce')
            df.dropna(subset=['season_end_year'], inplace=True)
            if not df['season_end_year'].empty:
                oldest_season_end_year = int(df['season_end_year'].min())
                newest_season_end_year = int(df['season_end_year'].max())
        
        if oldest_season_end_year is None and 'season' in df.columns:
            df.dropna(subset=['season'], inplace=True)
            
            def extract_end_year_from_season_str(season_str):
                if isinstance(season_str, str) and '-' in season_str:
                    parts = season_str.split('-')
                    if len(parts) == 2:
                        start_year_str, end_part_str = parts
                        if start_year_str.isdigit() and end_part_str.isdigit():
                            start_year = int(start_year_str)
                            if len(end_part_str) == 2: 
                                return start_year + 1
                            elif len(end_part_str) == 4: 
                                return int(end_part_str)
                return None

            df['calc_season_end_year'] = df['season'].apply(extract_end_year_from_season_str)
            df.dropna(subset=['calc_season_end_year'], inplace=True)
            if not df['calc_season_end_year'].empty:
                oldest_season_end_year = int(df['calc_season_end_year'].min())
                newest_season_end_year = int(df['calc_season_end_year'].max())
        
        if oldest_season_end_year and newest_season_end_year:
            return oldest_season_end_year, newest_season_end_year
        else:
            return None, None
    except Exception:
        return None, None

# Downloads and parses game statistics data for a single season from Bart Torvik
def fetch_season_gamelogs_bart_torvik_full(year):
    season_url = BASE_URL_PATTERN.format(year=year)
    try:
        response = requests.get(season_url, headers=HEADERS, timeout=60) 
        response.raise_for_status() 
        csv_content = response.text
        try:
            df_season = pd.read_csv(io.StringIO(csv_content), header=None, 
                                    names=EXPECTED_BART_COLUMN_NAMES_FULL, 
                                    usecols=range(NUM_EXPECTED_COLS_FULL), 
                                    na_filter=False, dtype=str)
        except pd.errors.ParserError:
            return pd.DataFrame()
        except Exception:
             return pd.DataFrame()

        start_year_of_season = year - 1
        end_year_short = str(year)[-2:] 
        df_season['season_label'] = f"{start_year_of_season}-{end_year_short}" 
        df_season['season_end_year'] = year
        return df_season
    except requests.exceptions.HTTPError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Fetches data for multiple seasons and combines into a single cleaned dataframe
def load_bart_data_full(oldest_season, newest_season):
    all_bart_team_game_stats_list = []
    for year in range(oldest_season, newest_season + 1):
        season_df = fetch_season_gamelogs_bart_torvik_full(year)
        if not season_df.empty:
            all_bart_team_game_stats_list.append(season_df)
        if year < newest_season: 
            time.sleep(REQUEST_DELAY_SECONDS)

    if not all_bart_team_game_stats_list:
        return pd.DataFrame()

    bart_df = pd.concat(all_bart_team_game_stats_list, ignore_index=True)

    if 'bart_date_raw' in bart_df.columns and 'season_end_year' in bart_df.columns:
        def assign_full_date_from_gamelog(row):
            try:
                date_parts = str(row['bart_date_raw']).split('/')
                month = int(date_parts[0])
                day = int(date_parts[1])
                year_from_date_col_str = str(date_parts[2])
                year_from_date_col = int(year_from_date_col_str) + 2000 if len(year_from_date_col_str) == 2 else int(year_from_date_col_str)
                
                season_end_yr = int(row['season_end_year'])
                actual_year = (season_end_yr - 1) if month >= 8 else season_end_yr
                
                if year_from_date_col and abs(year_from_date_col - actual_year) <= 1:
                    pass
                else:
                    pass
                return pd.Timestamp(f"{actual_year}-{month}-{day}")
            except Exception:
                return pd.NaT
        bart_df['game_date'] = bart_df.apply(assign_full_date_from_gamelog, axis=1)
        bart_df.dropna(subset=['game_date'], inplace=True)
        bart_df['game_date_str'] = bart_df['game_date'].dt.strftime('%Y-%m-%d')
        
    numeric_cols_bart = [
        'bart_adj_o_ingame', 'bart_adj_d_ingame', 'bart_tempo_ingame',
        'bart_off_eff_ingame', 'bart_off_efg_pct_ingame', 'bart_off_to_pct_ingame', 
        'bart_off_orb_pct_ingame', 'bart_off_ftr_ingame',
        'bart_def_eff_ingame', 'bart_def_efg_pct_ingame', 
        'bart_def_to_pct_ingame', 'bart_def_drb_pct_ingame', 'bart_def_ftr_ingame',
        'bart_opponent_conf_order', 'bart_season_col', 'bart_possessions_game',
        'bart_proj_score_diff', 'bart_pred_barthag_wp', 'bart_row_id_or_flag'
    ]
    for col in numeric_cols_bart:
        if col in bart_df.columns:
            bart_df[col] = pd.to_numeric(bart_df[col], errors='coerce')
            
    return bart_df

def main():
    os.makedirs(OUTPUT_DIR_RAW, exist_ok=True)

    OLDEST_SEASON_END_YEAR, NEWEST_SEASON_END_YEAR = get_season_range_from_sbr_data_content(SBR_ELO_DATA_PATH)
    if OLDEST_SEASON_END_YEAR is None or NEWEST_SEASON_END_YEAR is None:
        return
    
    bart_data_full_df = load_bart_data_full(OLDEST_SEASON_END_YEAR, NEWEST_SEASON_END_YEAR)

    if not bart_data_full_df.empty:
        try:
            bart_data_full_df.to_csv(BART_TORVIK_FULL_OUTPUT_CSV, index=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()
