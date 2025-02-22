# Adds cumulative coaching statistics and performance metrics to game features
import pandas as pd
import numpy as np
import os

FEATURES_PATH = os.path.join("data", "features", "ncaab_cumulative_features_v4_ref_opeid_ema.csv")
COACH_PATH = os.path.join("data", "features", "coach_sr_true_starts_and_stats.csv")                              
OUTPUT_PATH = os.path.join("data", "features", "ncaab_features_with_coach_stats_v8_seasonal_ema.csv")              

EMA_ALPHA = 0.2

CUMULATIVE_FEATURES = [
    'pg_coach_games_at_team',
    'pg_coach_seasons_at_team',
    'pg_coach_is_first_season_at_team',
    'pg_coach_wins_at_team',
    'pg_coach_losses_at_team',
    'pg_coach_win_pct_at_team',
    'pg_coach_games_this_season',
    'pg_coach_wins_this_season',
    'pg_coach_win_pct_this_season'
]

EMA_FEATURES = [
    'pg_coach_ema_win_pct_this_season'
]

# Extracts the starting year from season string format
def parse_season_year(season_str):
    if pd.isna(season_str) or not isinstance(season_str, str) or '-' not in season_str:
        return None
    try:
        return int(season_str.split('-')[0])
    except ValueError:
        return None

# Processes game data to calculate cumulative coaching statistics and EMA metrics
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        print(f"Main game data file not found: {FEATURES_PATH}")
        return
    
    main_df = pd.read_csv(FEATURES_PATH, low_memory=False)
    main_df['date'] = pd.to_datetime(main_df['date'])
    main_df.sort_values(by=['season', 'date'], inplace=True)
    main_df.reset_index(drop=True, inplace=True)

    if not os.path.exists(COACH_PATH):
        print(f"Coach history file not found: {COACH_PATH}")
        return
    
    coach_df = pd.read_csv(COACH_PATH)
    coach_df['coach_true_start_year_at_team'] = coach_df['true_start_season_sr'].apply(parse_season_year)
    
    coach_history = {}
    for _, row in coach_df.iterrows():
        key = (row['coach_name'], row['team_name'])
        coach_history[key] = {
            'games_before': row.get('games_before_dataset_at_team', 0),
            'wins_before': row.get('wins_before_dataset_at_team', 0),
            'losses_before': row.get('losses_before_dataset_at_team', 0),
            'true_start_year': row['coach_true_start_year_at_team']
        }

    for prefix in ['visitor_', 'home_']:
        for feature in CUMULATIVE_FEATURES + EMA_FEATURES:
            main_df[f"{prefix}{feature}"] = np.nan

    overall_stats = {}
    season_stats = {}

    for index, row in main_df.iterrows():
        current_season_year = parse_season_year(row['season'])

        for perspective in ['visitor', 'home']:
            prefix = f'{perspective}_'
            coach_name = row.get(f'{prefix}bart_team_coach')
            team_name = row.get(f'{prefix}team')
            team_score = row.get(f'{prefix}final_score')
            
            opponent_score_col = 'home_final_score' if perspective == 'visitor' else 'visitor_final_score'
            opponent_score = row.get(opponent_score_col)

            if pd.isna(coach_name) or pd.isna(team_name) or pd.isna(team_score) or pd.isna(opponent_score):
                continue

            coach_team_key = (coach_name, team_name)
            prior = coach_history.get(coach_team_key, {
                'games_before': 0, 'wins_before': 0, 'losses_before': 0, 'true_start_year': None
            })
            games_before = prior['games_before']
            wins_before = prior['wins_before']
            losses_before = prior['losses_before']
            coach_start_year = prior['true_start_year']

            overall_key = (coach_name, team_name)
            current_overall = overall_stats.get(overall_key, {'games': 0, 'wins': 0, 'losses': 0})
            games_overall_prior = current_overall['games']
            wins_overall_prior = current_overall['wins']
            losses_overall_prior = current_overall['losses']

            season_key = (coach_name, team_name, current_season_year)
            current_season = season_stats.get(season_key, {'games': 0, 'wins': 0, 'current_ema_this_season': np.nan})
            games_season_prior = current_season['games']
            wins_season_prior = current_season['wins']
            ema_val = current_season['current_ema_this_season']

            total_games = games_before + games_overall_prior
            main_df.loc[index, f"{prefix}pg_coach_games_at_team"] = total_games

            if coach_start_year is not None and current_season_year is not None:
                seasons_at_team = current_season_year - coach_start_year + 1
                main_df.loc[index, f"{prefix}pg_coach_seasons_at_team"] = seasons_at_team
                main_df.loc[index, f"{prefix}pg_coach_is_first_season_at_team"] = 1 if seasons_at_team <= 1 else 0
            else:
                is_first = (games_before == 0 and games_overall_prior == 0)
                main_df.loc[index, f"{prefix}pg_coach_seasons_at_team"] = 1 if is_first else np.nan
                main_df.loc[index, f"{prefix}pg_coach_is_first_season_at_team"] = 1 if is_first else 0

            total_wins = wins_before + wins_overall_prior
            main_df.loc[index, f"{prefix}pg_coach_wins_at_team"] = total_wins
            total_losses = losses_before + losses_overall_prior
            main_df.loc[index, f"{prefix}pg_coach_losses_at_team"] = total_losses

            if total_games > 0:
                main_df.loc[index, f"{prefix}pg_coach_win_pct_at_team"] = total_wins / total_games
            else:
                main_df.loc[index, f"{prefix}pg_coach_win_pct_at_team"] = 0.0

            main_df.loc[index, f"{prefix}pg_coach_games_this_season"] = games_season_prior
            main_df.loc[index, f"{prefix}pg_coach_wins_this_season"] = wins_season_prior
            if games_season_prior > 0:
                main_df.loc[index, f"{prefix}pg_coach_win_pct_this_season"] = wins_season_prior / games_season_prior
            else:
                main_df.loc[index, f"{prefix}pg_coach_win_pct_this_season"] = 0.0

            main_df.loc[index, f"{prefix}pg_coach_ema_win_pct_this_season"] = ema_val

            game_result = 1.0 if team_score > opponent_score else 0.0

            overall_update = overall_stats.setdefault(overall_key, {'games': 0, 'wins': 0, 'losses': 0})
            overall_update['games'] += 1
            if game_result == 1.0: 
                overall_update['wins'] += 1
            else: 
                overall_update['losses'] += 1
            
            season_update = season_stats.setdefault(season_key, {'games': 0, 'wins': 0, 'current_ema_this_season': np.nan})
            season_update['games'] += 1
            if game_result == 1.0: 
                season_update['wins'] += 1

            prev_ema = ema_val
            if pd.isna(prev_ema):
                season_update['current_ema_this_season'] = game_result
            else:
                season_update['current_ema_this_season'] = EMA_ALPHA * game_result + (1 - EMA_ALPHA) * prev_ema

    try:
        main_df.to_csv(OUTPUT_PATH, index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    main()