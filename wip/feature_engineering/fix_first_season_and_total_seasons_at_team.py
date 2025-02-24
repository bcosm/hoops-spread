# Corrects coach tenure calculations using Sports Reference start season data
import pandas as pd
import numpy as np
import os

MAIN_CSV = os.path.join("data", "features", "ncaab_features_with_coach_stats_v8_seasonal_ema.csv")
COACH_CSV = os.path.join("data", "features", "coach_sr_true_starts_and_stats.csv")
OUTPUT_CSV = os.path.join("data", "features", "ncaab_cumulative_features_v9_fixed_coach_seasons_debug.csv")

# Extracts starting year from season string format
def parse_season(season_str):
    if pd.isna(season_str) or not isinstance(season_str, str) or '-' not in season_str:
        return None
    try:
        return int(season_str.split('-')[0])
    except ValueError:
        return None

# Updates coach seasons and first season flags using corrected start dates
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    if not os.path.exists(MAIN_CSV):
        return
    
    try:
        main_df = pd.read_csv(MAIN_CSV, low_memory=False)
        main_df.sort_values(by=['season', 'date'], inplace=True)
        main_df.reset_index(drop=True, inplace=True)
    except Exception:
        return

    if main_df.empty:
        return

    coach_map = {}
    if os.path.exists(COACH_CSV):
        try:
            coach_df = pd.read_csv(COACH_CSV)
            for _, row in coach_df.iterrows():
                coach_name = row.get('coach_name')
                team_name = row.get('team_name')
                start_season = row.get('true_start_season_sr')
                start_year = parse_season(start_season)
                
                if (pd.notna(coach_name) and pd.notna(team_name) and start_year is not None and
                    isinstance(coach_name, str) and coach_name.strip() and
                    isinstance(team_name, str) and team_name.strip()):
                    coach_map[(coach_name, team_name)] = start_year
        except Exception:
            coach_map = {}

    for prefix in ['visitor_', 'home_']:
        main_df[f"{prefix}pg_coach_seasons_at_team"] = np.nan
        main_df[f"{prefix}pg_coach_is_first_season_at_team"] = np.nan

    first_season_map = {}

    for index, row in main_df.iterrows():
        current_year = parse_season(row['season'])
        if current_year is None:
            continue

        for perspective in ['visitor', 'home']:
            prefix = f'{perspective}_'
            coach_name = row.get(f'{prefix}bart_team_coach')
            team_name = row.get(f'{prefix}team')

            if (pd.isna(coach_name) or pd.isna(team_name) or
                not isinstance(coach_name, str) or not coach_name.strip() or
                not isinstance(team_name, str) or not team_name.strip()):
                continue

            coach_team_key = (coach_name, team_name)
            
            if coach_team_key in coach_map:
                start_year = coach_map[coach_team_key]
            else:
                if coach_team_key in first_season_map:
                    start_year = first_season_map[coach_team_key]
                else:
                    start_year = current_year
                    first_season_map[coach_team_key] = start_year

            if start_year is not None:
                seasons_count = current_year - start_year + 1
                if seasons_count >= 1:
                    main_df.loc[index, f"{prefix}pg_coach_seasons_at_team"] = seasons_count
                    main_df.loc[index, f"{prefix}pg_coach_is_first_season_at_team"] = 1.0 if seasons_count == 1 else 0.0

    try:
        main_df.to_csv(OUTPUT_CSV, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()