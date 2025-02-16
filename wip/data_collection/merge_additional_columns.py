# Merges additional Bart Torvik columns with main game data using school codes
import os
import pandas as pd
import json

MAIN_DATA_PATH = os.path.join("data", "processed_with_school_codes", "merged_sbr_elo_bart_torvik_games_fully_cohesive.csv")
BART_DATA_PATH = os.path.join("data", "raw", "bart_torvik_full_raw_data.csv") 
SCHOOL_CODE_PATH = os.path.join("data", "processed_with_school_codes", "llm_school_name_to_code.json")

OUTPUT_DIR = os.path.join("data", "processed") 
FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, "sbr_elo_bart_with_extra_cols_v4.csv")
FAILURES_CSV = os.path.join(OUTPUT_DIR, "diagnostics_opeid_lookup_failures_v4.csv")

EXTRA_BART_COLS = [
    'bart_opponent_conf_abb', 'bart_opponent_conf_order', 'bart_season_col',
    'bart_possessions_game', 'bart_game_id_str', 'bart_team_coach',
    'bart_opponent_coach', 'bart_proj_score_diff', 'bart_pred_barthag_wp',
    'bart_box_score_array_str', 'bart_row_id_or_flag'
]
GAME_LEVEL_COLS = ['bart_possessions_game', 'bart_game_id_str', 'bart_box_score_array_str', 'bart_season_col', 'bart_row_id_or_flag']
TEAM_SPECIFIC_COLS = [col for col in EXTRA_BART_COLS if col not in GAME_LEVEL_COLS]


# Gets school OPEID from team name using exact and lowercase matching
def get_opeid(team_name, school_map, school_map_lower):
    if pd.isna(team_name) or not isinstance(team_name, str):
        return None
    
    opeid = school_map.get(team_name) 
    if opeid is not None:
        return str(opeid).split('.')[0]

    opeid = school_map_lower.get(team_name.lower())
    if opeid is not None:
        return str(opeid).split('.')[0]
    
    return None

# Main function to merge additional columns from Bart Torvik data using school code matching
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SCHOOL_CODE_PATH):
        return
    try:
        with open(SCHOOL_CODE_PATH, 'r') as f:
            school_map = json.load(f) 
        school_map_lower = {k.lower(): v for k, v in school_map.items()}
    except Exception as e:
        return

    if not os.path.exists(MAIN_DATA_PATH):
        return
    try:
        main_df = pd.read_csv(MAIN_DATA_PATH, low_memory=False, dtype={'visitor_rot': str, 'home_rot': str})
        main_df['game_date_str'] = pd.to_datetime(main_df['date']).dt.strftime('%Y-%m-%d')
        if 'season_label' not in main_df.columns:
            if 'season' in main_df.columns:
                 # Converts SBR season format to standardized label format
                 def sbr_season_to_label(season_str):
                    if isinstance(season_str, str) and '-' in season_str:
                        parts = season_str.split('-');
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            return f"{parts[0]}-{parts[1][-2:]}"
                    return None
                 main_df['season_label'] = main_df['season'].apply(sbr_season_to_label)
            else: 
                return
        main_df.dropna(subset=['season_label', 'game_date_str'], inplace=True)
    except Exception as e:
        return

    if not os.path.exists(BART_DATA_PATH):
        return
    try:
        bart_df = pd.read_csv(BART_DATA_PATH, dtype=str, low_memory=False)
        if 'game_date_str' not in bart_df.columns or 'season_label' not in bart_df.columns:
            return
    except Exception as e:
        return

    failures = []
    main_df['sbr_visitor_opeid'] = main_df['visitor_team'].astype(str).apply(lambda x: get_opeid(x, school_map, school_map_lower) or failures.append({'team_name': x, 'source_df': 'main_df', 'role': 'visitor'}))
    main_df['sbr_home_opeid'] = main_df['home_team'].astype(str).apply(lambda x: get_opeid(x, school_map, school_map_lower) or failures.append({'team_name': x, 'source_df': 'main_df', 'role': 'home'}))
    
    bart_df['bart_team_opeid'] = bart_df['bart_team'].astype(str).apply(lambda x: get_opeid(x, school_map, school_map_lower) or failures.append({'team_name': x, 'source_df': 'bart_df', 'role': 'bart_team'}))

    if failures:
        failures_df = pd.DataFrame(failures).drop_duplicates().sort_values(by=['source_df', 'role', 'team_name'])
        failures_df.to_csv(FAILURES_CSV, index=False)

    main_df_with_opeids = main_df.dropna(subset=['sbr_visitor_opeid', 'sbr_home_opeid']).copy()
    bart_df_with_opeids = bart_df.dropna(subset=['bart_team_opeid']).copy() 
    if main_df_with_opeids.empty or bart_df_with_opeids.empty:
        return

    bart_cols_to_select = ['game_date_str', 'season_label', 'bart_team_opeid'] + EXTRA_BART_COLS
    bart_df_subset = bart_df_with_opeids[[col for col in bart_cols_to_select if col in bart_df_with_opeids.columns]].copy()
    bart_df_subset = bart_df_subset.drop_duplicates(subset=['game_date_str', 'season_label', 'bart_team_opeid'], keep='first')

    visitor_rename_map = {'bart_team_opeid': 'sbr_visitor_opeid'}
    for col in TEAM_SPECIFIC_COLS:
        visitor_rename_map[col] = f"visitor_{col}"
    merged_df_final = pd.merge(
        main_df_with_opeids,
        bart_df_subset.rename(columns=visitor_rename_map),
        on=['game_date_str', 'season_label', 'sbr_visitor_opeid'],
        how='left',
        suffixes=('', '_vis_extra_dup')
    )
    for col in GAME_LEVEL_COLS:
        if f"{col}_vis_extra_dup" in merged_df_final.columns:
            if col not in main_df_with_opeids.columns:
                 merged_df_final.rename(columns={f"{col}_vis_extra_dup": col}, inplace=True)
            else:
                merged_df_final[col] = merged_df_final[f"{col}_vis_extra_dup"].fillna(merged_df_final[col])
                merged_df_final.drop(columns=[f"{col}_vis_extra_dup"], inplace=True)

    home_rename_map = {'bart_team_opeid': 'sbr_home_opeid'}
    home_cols = ['game_date_str', 'season_label', 'bart_team_opeid']
    for col in TEAM_SPECIFIC_COLS:
        home_rename_map[col] = f"home_{col}"
        home_cols.append(col) 
    
    home_cols = [c for c in home_cols if c in bart_df_subset.columns]
    home_cols = sorted(list(set(home_cols)), key=home_cols.index)

    merged_df_final = pd.merge(
        merged_df_final,
        bart_df_subset[home_cols].rename(columns=home_rename_map).drop_duplicates(subset=['game_date_str', 'season_label', 'sbr_home_opeid']),
        on=['game_date_str', 'season_label', 'sbr_home_opeid'],
        how='left',
        suffixes=('', '_home_extra_dup')                                                                           
    )
    
    cols_to_drop = [col for col in merged_df_final.columns if '_home_extra_dup' in col]
    merged_df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    merged_df_final.drop(columns=['sbr_visitor_opeid', 'sbr_home_opeid'], inplace=True, errors='ignore')

    game_keys = ['date', 'visitor_team', 'home_team'] 
    if all(key_col in merged_df_final.columns for key_col in game_keys):
        merged_df_final.drop_duplicates(subset=game_keys, keep='first', inplace=True)

    try:
        merged_df_final.to_csv(FINAL_CSV_PATH, index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()
