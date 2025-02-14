# Merges SBR game data with Bart Torvik team statistics using school codes
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed_with_school_codes")
SBR_INPUT_CSV = PROCESSED_DIR / "sbr_elo_with_llm_codes.csv"
BART_INPUT_CSV = PROCESSED_DIR / "bart_torvik_with_llm_codes.csv"
MERGED_OUTPUT_CSV = PROCESSED_DIR / "merged_sbr_elo_bart_torvik_games_fully_cohesive.csv"           

# Combines SBR betting data with Bart Torvik statistics for both visitor and home teams
def merge_datasets():
    try:
        sbr_df = pd.read_csv(SBR_INPUT_CSV)
    except FileNotFoundError:
        print(f"SBR/Elo input file not found at {SBR_INPUT_CSV}")
        return
    except Exception as e:
        print(f"Could not read SBR/Elo CSV: {e}")
        return

    try:
        bart_df = pd.read_csv(BART_INPUT_CSV)
    except FileNotFoundError:
        print(f"Bart Torvik input file not found at {BART_INPUT_CSV}")
        return
    except Exception as e:
        print(f"Could not read Bart Torvik CSV: {e}")
        return

    try:
        sbr_df['date'] = pd.to_datetime(sbr_df['date'])
        try:
            bart_df['bart_date_raw'] = pd.to_datetime(bart_df['bart_date_raw'], format='%m/%d/%y')
        except ValueError:
            bart_df['bart_date_raw'] = pd.to_datetime(bart_df['bart_date_raw'])
        bart_df.rename(columns={'bart_date_raw': 'date'}, inplace=True)
    except Exception as e:
        print(f"Could not convert date columns: {e}")
        return

    sbr_df = sbr_df.reset_index().rename(columns={'index': 'sbr_game_id'})
    initial_count = len(sbr_df)

    bart_df.drop_duplicates(subset=['date', 'team_federal_code'], keep='first', inplace=True)

    bart_stat_cols = [
        col for col in bart_df.columns
        if col not in [
            'date', 'bart_game_type', 'bart_team', 'bart_team_conf', 
            'bart_opponent', 'bart_venue', 'bart_result_string',
            'team_federal_code', 'opponent_federal_code',
            'team_original_sbr_school_name', 'opponent_original_sbr_school_name',
            'season_label', 'season_end_year'
        ]
    ]

    visitor_df = bart_df[['date', 'team_federal_code'] + bart_stat_cols].copy()
    visitor_rename = {'team_federal_code': 'visitor_federal_code_bart'}
    for col in bart_stat_cols: 
        visitor_rename[col] = f'visitor_{col}_bart'
    visitor_df.rename(columns=visitor_rename, inplace=True)

    home_df = bart_df[['date', 'team_federal_code'] + bart_stat_cols].copy()
    home_rename = {'team_federal_code': 'home_federal_code_bart'}
    for col in bart_stat_cols: 
        home_rename[col] = f'home_{col}_bart'
    home_df.rename(columns=home_rename, inplace=True)

    merged_df = pd.merge(
        sbr_df,
        visitor_df,
        left_on=['date', 'visitor_federal_code'],
        right_on=['date', 'visitor_federal_code_bart'],
        how='left'
    )
    if 'visitor_federal_code_bart' in merged_df.columns:
        merged_df.drop(columns=['visitor_federal_code_bart'], inplace=True)
    final_df = pd.merge(
        merged_df,
        home_df,
        left_on=['date', 'home_federal_code'],
        right_on=['date', 'home_federal_code_bart'],
        how='left',
        suffixes=('_leftDefault', '_home_conflict')
    )
    if 'home_federal_code_bart' in final_df.columns:
        final_df.drop(columns=['home_federal_code_bart'], inplace=True)
    
    if len(final_df) > initial_count:
        final_df.drop_duplicates(subset=['sbr_game_id'], keep='first', inplace=True)
    
    visitor_cols = [f'visitor_{col}_bart' for col in bart_stat_cols]
    home_cols = [f'home_{col}_bart' for col in bart_stat_cols]
    all_bart_cols = visitor_cols + home_cols
    
    existing_cols = [col for col in all_bart_cols if col in final_df.columns]
    if existing_cols:
        final_df.dropna(subset=existing_cols, how='any', inplace=True)

    try:
        if 'sbr_game_id' in final_df.columns:
            final_df.drop(columns=['sbr_game_id'], inplace=True)
            
        final_df.to_csv(MERGED_OUTPUT_CSV, index=False)
        print(f"Merged data saved to: {MERGED_OUTPUT_CSV}")
        print(f"Final dataset contains {len(final_df)} games")
    except Exception as e:
        print(f"Could not save merged CSV: {e}")
if __name__ == "__main__":
    merge_datasets()