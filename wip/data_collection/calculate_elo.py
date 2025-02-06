# Calculate Elo ratings for college basketball teams based on game results and margins
import os
import pandas as pd
import numpy as np
from datetime import datetime

SBR_INPUT_CSV_PATH = os.path.join("data", "raw", "sbr_ncaab_2007_2022.csv")
OUTPUT_DIR = os.path.join("data", "processed")
ELO_AUGMENTED_SBR_CSV = os.path.join(OUTPUT_DIR, "sbr_ncaab_with_elo_ratings.csv")

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE_ELO = 65

USE_ELO_MARGIN_MULTIPLIER = True
ELO_MARGIN_C = 1                                                                                                                 

# Calculate expected win probability for team A against team B using Elo ratings
def calculate_expected_score(elo_a, elo_b):
    return 1 / (1 + 10**((elo_b - elo_a) / 400))

# Get margin of victory multiplier to adjust Elo rating changes based on game closeness
def get_elo_margin_multiplier(margin_of_victory, elo_diff_winner_loser):    
    if not USE_ELO_MARGIN_MULTIPLIER:
        return 1.0
    return np.log(abs(margin_of_victory) + 1) * (2.2 / ((elo_diff_winner_loser * 0.001) + 2.2))

# Update Elo ratings for winner and loser based on margin of victory
def update_elo(elo_winner, elo_loser, margin_of_victory):
    expected_winner = calculate_expected_score(elo_winner, elo_loser)
    elo_change = K_FACTOR * (1 - expected_winner)
    
    if USE_ELO_MARGIN_MULTIPLIER:
        multiplier = get_elo_margin_multiplier(margin_of_victory, elo_winner - elo_loser)
        elo_change *= multiplier
        
    new_elo_winner = elo_winner + elo_change
    new_elo_loser = elo_loser - elo_change
    return new_elo_winner, new_elo_loser

# Process a single season's games to calculate pre-game Elo ratings for each team
def process_season_for_elo(season_df, initial_elo, k_factor, home_advantage_elo):
    teams_in_season = pd.concat([season_df['visitor_team'], season_df['home_team']]).astype(str).unique()
    current_elos = {team: initial_elo for team in teams_in_season}
    
    pre_game_elos_visitor = []
    pre_game_elos_home = []
    
    season_df['date'] = pd.to_datetime(season_df['date'])
    if 'visitor_rot' in season_df.columns:
        season_df.sort_values(by=['date', 'visitor_rot'], inplace=True)
    else:
        season_df.sort_values(by=['date'], inplace=True)

    for index, row in season_df.iterrows():
        visitor_team = str(row['visitor_team'])
        home_team = str(row['home_team'])
        
        elo_visitor_pregame = current_elos.get(visitor_team, initial_elo)
        elo_home_pregame_no_hca = current_elos.get(home_team, initial_elo)
        
        pre_game_elos_visitor.append(elo_visitor_pregame)
        pre_game_elos_home.append(elo_home_pregame_no_hca)

        elo_home_with_hca = elo_home_pregame_no_hca + home_advantage_elo
        
        visitor_score = row['visitor_final_score']
        home_score = row['home_final_score']
        
        if pd.isna(visitor_score) or pd.isna(home_score):
            continue

        margin = abs(home_score - visitor_score)

        if home_score > visitor_score:
            new_elo_home, new_elo_visitor = update_elo(elo_home_with_hca, elo_visitor_pregame, margin)
            current_elos[home_team] = new_elo_home - home_advantage_elo
            current_elos[visitor_team] = new_elo_visitor
        elif visitor_score > home_score:
            new_elo_visitor, new_elo_home = update_elo(elo_visitor_pregame, elo_home_with_hca, margin)
            current_elos[visitor_team] = new_elo_visitor
            current_elos[home_team] = new_elo_home - home_advantage_elo

    season_df['visitor_pregame_elo'] = pre_game_elos_visitor
    season_df['home_pregame_elo'] = pre_game_elos_home
    
    return season_df

# Main function to load data, calculate Elo ratings by season, and save results
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SBR_INPUT_CSV_PATH):
        return
    
    try:
        sbr_df = pd.read_csv(SBR_INPUT_CSV_PATH)
        sbr_df['visitor_final_score'] = pd.to_numeric(sbr_df['visitor_final_score'], errors='coerce')
        sbr_df['home_final_score'] = pd.to_numeric(sbr_df['home_final_score'], errors='coerce')
    except Exception:
        return

    all_seasons_augmented_list = []
    
    if 'season' not in sbr_df.columns:
        return

    for season, season_games_df in sbr_df.groupby('season'):
        season_games_df_copy = season_games_df.copy()
        augmented_season_df = process_season_for_elo(
            season_games_df_copy, 
            INITIAL_ELO, 
            K_FACTOR, 
            HOME_ADVANTAGE_ELO
        )
        all_seasons_augmented_list.append(augmented_season_df)

    if not all_seasons_augmented_list:
        return

    final_df = pd.concat(all_seasons_augmented_list)
    
    final_df['elo_diff_home_adv'] = (final_df['home_pregame_elo'] + HOME_ADVANTAGE_ELO) - final_df['visitor_pregame_elo']
    final_df['elo_diff_raw'] = final_df['home_pregame_elo'] - final_df['visitor_pregame_elo']

    try:
        final_df.to_csv(ELO_AUGMENTED_SBR_CSV, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()
