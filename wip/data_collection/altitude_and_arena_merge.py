# Merge arena location data with altitude information using external elevation API for NCAAB games
import os
import pandas as pd
import json
import requests
import time
import re
import numpy as np

MAIN_DATA_CSV_PATH = os.path.join("data", "processed", "ncaab_data_cleaned.csv")
ARENA_DATA_CSV_PATH = os.path.join("data", "processed", "sbr_ncaab_with_arena_locations.csv") 

OUTPUT_DIR_EXTERNAL = os.path.join("data", "external") 
OUTPUT_DIR_PROCESSED = os.path.join("data", "processed") 
ARENA_WITH_ALTITUDE_CSV_PATH = os.path.join(OUTPUT_DIR_EXTERNAL, "ncaab_arenas_with_altitude.csv")
FINAL_MERGED_CSV_PATH = os.path.join(OUTPUT_DIR_PROCESSED, "sbr_elo_bart_arena_direct_merge.csv")

ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
API_BATCH_SIZE = 50  
API_REQUEST_DELAY = 1  

FORCE_REFETCH_ALTITUDES = False 

ARENA_COLS_TO_MERGE = [
    'arena_name', 'city', 'state', 
    'latitude', 'longitude', 'geocoded_address',
    'wiki_team_name', 'match_score' 
]
ARENA_TEAM_NAME_COL = 'sbr_team_name'

# Fetch altitude data from elevation API in batches to avoid rate limiting
def get_altitudes_batched(lat_lon_pairs):
    all_results_ordered = [None] * len(lat_lon_pairs) 

    for i in range(0, len(lat_lon_pairs), API_BATCH_SIZE):
        batch = lat_lon_pairs[i:i + API_BATCH_SIZE]
        valid_batch_pairs = [(lat, lon) for lat, lon in batch if pd.notna(lat) and pd.notna(lon)]
        
        if not valid_batch_pairs:
            for j in range(len(batch)): 
                if i + j < len(all_results_ordered):
                    all_results_ordered[i+j] = None 
            continue

        locations_param = "|".join([f"{lat},{lon}" for lat, lon in valid_batch_pairs])
        
        print(f"Fetching altitude for batch {i // API_BATCH_SIZE + 1} ({len(valid_batch_pairs)} valid locations)...")
        try:
            response = requests.get(ELEVATION_API_URL, params={'locations': locations_param}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            api_results = data.get('results', [])
            
            current_api_result_idx = 0
            for k_batch_orig_idx in range(len(batch)):
                original_list_idx = i + k_batch_orig_idx
                lat, lon = batch[k_batch_orig_idx]
                if pd.notna(lat) and pd.notna(lon): 
                    if current_api_result_idx < len(api_results):
                        all_results_ordered[original_list_idx] = api_results[current_api_result_idx].get('elevation')
                        current_api_result_idx += 1
                    else: 
                        all_results_ordered[original_list_idx] = None
                else: 
                    all_results_ordered[original_list_idx] = None
            
            if current_api_result_idx != len(api_results) and len(valid_batch_pairs) == len(api_results):
                 print(f"Warning: Mismatch processing API results for batch {i // API_BATCH_SIZE + 1}.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching altitude for batch: {e}")
            for k_batch_idx in range(len(batch)): 
                original_list_idx = i + k_batch_idx
                if original_list_idx < len(all_results_ordered):
                    all_results_ordered[original_list_idx] = None
        except json.JSONDecodeError:
            print(f"Error decoding JSON response for batch.")
            for k_batch_idx in range(len(batch)):
                original_list_idx = i + k_batch_idx
                if original_list_idx < len(all_results_ordered):
                    all_results_ordered[original_list_idx] = None
        
        if i + API_BATCH_SIZE < len(lat_lon_pairs): 
            time.sleep(API_REQUEST_DELAY)
            
    return all_results_ordered


# Main function to orchestrate arena data loading, altitude fetching, and merging with game data
def main():
    os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_EXTERNAL, exist_ok=True)

    print(f"Loading arena data from: {ARENA_DATA_CSV_PATH}")
    if not os.path.exists(ARENA_DATA_CSV_PATH):
        print(f"ERROR: Arena data file not found: {ARENA_DATA_CSV_PATH}"); return
    try:
        arena_raw_df = pd.read_csv(ARENA_DATA_CSV_PATH)
        print(f"Loaded {len(arena_raw_df)} game entries with arena data.")
        
        visitor_arenas = arena_raw_df[['visitor_team', 'visitor_arena', 'visitor_city', 'visitor_state', 
                                     'visitor_latitude', 'visitor_longitude', 'visitor_geocoded_address']].copy()
        visitor_arenas.columns = ['sbr_team_name', 'arena_name', 'city', 'state', 
                                'latitude', 'longitude', 'geocoded_address']
        visitor_arenas['wiki_team_name'] = visitor_arenas['sbr_team_name']
        visitor_arenas['match_score'] = 1.0
        
        home_arenas = arena_raw_df[['home_team', 'home_arena', 'home_city', 'home_state', 
                                  'home_latitude', 'home_longitude', 'home_geocoded_address']].copy()
        home_arenas.columns = ['sbr_team_name', 'arena_name', 'city', 'state', 
                             'latitude', 'longitude', 'geocoded_address']
        home_arenas['wiki_team_name'] = home_arenas['sbr_team_name']
        home_arenas['match_score'] = 1.0
        
        arena_df = pd.concat([visitor_arenas, home_arenas], ignore_index=True)
        
        arena_df.drop_duplicates(subset=['sbr_team_name', 'arena_name'], keep='first', inplace=True)
        
        print(f"Reshaped to {len(arena_df)} unique team-arena combinations.")
    except Exception as e:
        print(f"Error loading arena data: {e}"); return

    if 'latitude' not in arena_df.columns or 'longitude' not in arena_df.columns:
        print("ERROR: Arena data CSV must contain 'latitude' and 'longitude' columns for altitude fetching.")
        if FORCE_REFETCH_ALTITUDES or not os.path.exists(ARENA_WITH_ALTITUDE_CSV_PATH):
            return
    
    if ARENA_TEAM_NAME_COL not in arena_df.columns:
        print(f"ERROR: Arena data CSV must contain the team name column: '{ARENA_TEAM_NAME_COL}'")
        return

    if FORCE_REFETCH_ALTITUDES or 'altitude_meters' not in arena_df.columns or arena_df['altitude_meters'].isnull().all():
        if 'latitude' in arena_df.columns and 'longitude' in arena_df.columns:
            print("Fetching altitudes for arenas...")
            arena_df['latitude'] = pd.to_numeric(arena_df['latitude'], errors='coerce')
            arena_df['longitude'] = pd.to_numeric(arena_df['longitude'], errors='coerce')
            
            lat_lon_pairs = list(zip(arena_df['latitude'], arena_df['longitude']))
            altitudes = get_altitudes_batched(lat_lon_pairs)
            arena_df['altitude_meters'] = altitudes
            
            print(f"Saving enriched arena data with altitudes to: {ARENA_WITH_ALTITUDE_CSV_PATH}")
            arena_df.to_csv(ARENA_WITH_ALTITUDE_CSV_PATH, index=False)
        else:
            print("Skipping altitude fetching as latitude/longitude columns are missing and not forcing refetch.")
    else:
        print(f"Loading already enriched arena data (with altitudes) from: {ARENA_WITH_ALTITUDE_CSV_PATH}")
        if os.path.exists(ARENA_WITH_ALTITUDE_CSV_PATH):
            arena_df = pd.read_csv(ARENA_WITH_ALTITUDE_CSV_PATH)
        else:
            print(f"Enriched arena file not found at {ARENA_WITH_ALTITUDE_CSV_PATH}. Please run with FORCE_REFETCH_ALTITUDES=True or ensure file exists.")
    
    if 'altitude_meters' not in arena_df.columns:
        print("Warning: 'altitude_meters' column is not present in arena_df. Altitude data will be missing.")
        arena_df['altitude_meters'] = np.nan

    print(f"\nLoading main synthesized data from: {MAIN_DATA_CSV_PATH}")
    if not os.path.exists(MAIN_DATA_CSV_PATH):
        print(f"ERROR: Main data file not found: {MAIN_DATA_CSV_PATH}"); return
    try:
        main_df = pd.read_csv(MAIN_DATA_CSV_PATH, low_memory=False, dtype={'visitor_rot': str, 'home_rot': str})
        if 'home_team' not in main_df.columns:
            print("ERROR: 'home_team' column not found in main dataset.")
            return
        print(f"Loaded {len(main_df)} games from main dataset.")
    except Exception as e:
        print(f"Error loading main data: {e}"); return

    cols_to_bring_from_arena = [ARENA_TEAM_NAME_COL] + ARENA_COLS_TO_MERGE + ['altitude_meters']
    cols_to_bring_from_arena_existing = [col for col in cols_to_bring_from_arena if col in arena_df.columns]
    arena_df_to_merge = arena_df[cols_to_bring_from_arena_existing].copy()
    
    arena_df_to_merge.drop_duplicates(subset=[ARENA_TEAM_NAME_COL], keep='first', inplace=True)
    print(f"Using {len(arena_df_to_merge)} unique arena entries for merge (based on '{ARENA_TEAM_NAME_COL}').")

    print(f"Merging all arena data with main dataset on main_df.home_team == arena_df.{ARENA_TEAM_NAME_COL}...")
    merged_df_final = pd.merge(
        main_df,
        arena_df_to_merge,
        left_on='home_team', 
        right_on=ARENA_TEAM_NAME_COL,
        how='left',
        suffixes=('', '_arena_dup')
    )
    
    if f"{ARENA_TEAM_NAME_COL}_arena_dup" in merged_df_final.columns:
         merged_df_final.drop(columns=[f"{ARENA_TEAM_NAME_COL}_arena_dup"], inplace=True, errors='ignore')
    elif ARENA_TEAM_NAME_COL in merged_df_final.columns and ARENA_TEAM_NAME_COL != 'home_team':
        merged_df_final.drop(columns=[ARENA_TEAM_NAME_COL], inplace=True, errors='ignore')

    for col_base in ARENA_COLS_TO_MERGE + ['altitude_meters']:
        suffixed_col = f"{col_base}_arena_dup"
        if suffixed_col in merged_df_final.columns:
            if col_base in main_df.columns:
                 merged_df_final[col_base] = merged_df_final[col_base].fillna(merged_df_final[suffixed_col])
            else:
                 merged_df_final.rename(columns={suffixed_col: col_base}, inplace=True)
            merged_df_final.drop(columns=[suffixed_col], inplace=True, errors='ignore')

    initial_row_count = len(merged_df_final)
    key_arena_info_col = 'latitude' 
    if key_arena_info_col in merged_df_final.columns: 
        merged_df_final.dropna(subset=[key_arena_info_col], inplace=True)
        rows_dropped = initial_row_count - len(merged_df_final)
        print(f"Dropped {rows_dropped} games due to missing essential arena information (based on '{key_arena_info_col}').")
    else:
        print(f"Warning: Key arena info column '{key_arena_info_col}' not found after merge. Cannot drop rows based on missing arena data using this key.")

    original_sbr_game_keys = ['date', 'visitor_team', 'home_team'] 
    if all(key_col in merged_df_final.columns for key_col in original_sbr_game_keys):
        rows_before_final_dedup = len(merged_df_final)
        merged_df_final.drop_duplicates(subset=original_sbr_game_keys, keep='first', inplace=True)
        print(f"Row count after final de-duplication on SBR game keys: {len(merged_df_final)} (dropped {rows_before_final_dedup - len(merged_df_final)})")
    else:
        print(f"Warning: Could not perform final de-duplication as one or more key columns are missing: {original_sbr_game_keys}")

    print(f"Merge of arena data complete. Resulting DataFrame has {len(merged_df_final)} rows.")

    print(f"\nSaving final dataset with all arena information to: {FINAL_MERGED_CSV_PATH}")
    try:
        merged_df_final.to_csv(FINAL_MERGED_CSV_PATH, index=False)
        print(f"Final dataset saved. Total games: {len(merged_df_final)}")
        if not merged_df_final.empty:
            print("\nSample of the data with new arena features (first 5 rows, selected columns):")
            sample_cols_to_show = ['date', 'visitor_team', 'home_team', 
                                   'arena_name', 'city', 'state', 'latitude', 'longitude', 
                                   'altitude_meters', 'wiki_team_name', 'match_score']
            sample_cols_to_show_existing = [col for col in sample_cols_to_show if col in merged_df_final.columns]
            print(merged_df_final[sample_cols_to_show_existing].head())
        else:
            print("Final DataFrame is empty.")
            
    except Exception as e:
        print(f"Error saving final CSV: {e}")

    print("\nProcess complete.")

if __name__ == "__main__":
    main()
