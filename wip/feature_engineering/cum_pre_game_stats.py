# Calculates cumulative pre-game statistics and EMA metrics for teams using historical data
import os
import numpy as np
import json
import re
import pandas as pd

INPUT_CSV = os.path.join("data", "processed", "sbr_elo_bart_arena_direct_merge.csv")
SCHOOL_CODE_JSON = os.path.join("data", "processed_with_school_codes", "llm_school_name_to_code.json")
OUTPUT_DIR = os.path.join("data", "features")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ncaab_cumulative_features_v4_ref_opeid_ema.csv")

EMA_ALPHA = 0.2

BART_STATS = [
    'adj_o_ingame', 'adj_d_ingame', 'tempo_ingame',
    'off_eff_ingame', 'off_efg_pct_ingame', 'off_to_pct_ingame',
    'off_orb_pct_ingame', 'off_ftr_ingame',
    'def_eff_ingame', 'def_efg_pct_ingame',
    'def_to_pct_ingame', 'def_drb_pct_ingame', 'def_ftr_ingame'
]

# Converts team name to standardized OPEID using exact and lowercase matching
def get_opeid(team_name, code_map, lower_map):
    if pd.isna(team_name) or not isinstance(team_name, str) or not team_name.strip():
        return None

    opeid = code_map.get(team_name)
    if opeid is not None:
        return str(opeid).split('.')[0].zfill(6)

    opeid = lower_map.get(team_name.lower())
    if opeid is not None:
        return str(opeid).split('.')[0].zfill(6)
    
    return None

# Parses box score JSON to extract FGA and 3PA for specific team
def parse_box_score(box_score_str, team_name, opp_name, code_map, lower_map):
    if pd.isna(box_score_str) or not isinstance(box_score_str, str):
        return np.nan, np.nan

    try:
        cleaned = str(box_score_str).strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        cleaned = cleaned.replace('\\"', '"')

        if not cleaned.startswith('[') or not cleaned.endswith(']'):
            return np.nan, np.nan

        stats = json.loads(cleaned)
        if not isinstance(stats, list) or len(stats) < 32:
            return np.nan, np.nan

        team_opeid = get_opeid(team_name, code_map, lower_map)
        
        team1_name = stats[2]
        team2_name = stats[3]

        team1_opeid = get_opeid(team1_name, code_map, lower_map)
        team2_opeid = get_opeid(team2_name, code_map, lower_map)

        if team_opeid is None:
            return np.nan, np.nan
            
        if team1_opeid is not None and team_opeid == team1_opeid:
            return int(stats[5]), int(stats[7])  # FGA, 3PA
        elif team2_opeid is not None and team_opeid == team2_opeid:
            return int(stats[19]), int(stats[21])  # FGA, 3PA
        else:
            opp_opeid = get_opeid(opp_name, code_map, lower_map)
            if opp_opeid is not None:
                if team1_opeid is not None and opp_opeid == team1_opeid:
                    return int(stats[19]), int(stats[21])  # Opponent's stats
                elif team2_opeid is not None and opp_opeid == team2_opeid:
                    return int(stats[5]), int(stats[7])  # Opponent's stats

        return np.nan, np.nan

    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        return np.nan, np.nan

# Calculates both simple averages and EMA values for team statistics from historical games
def calc_pregame_stats(history_df, prefix, alpha):
    stats = {}
    n_games = len(history_df)
    stats[f'{prefix}games_played_season'] = n_games

    if history_df.empty:
        for stat in BART_STATS:
            stats[f'{prefix}pregame_avg_{stat}'] = np.nan
            stats[f'{prefix}pregame_ema_{stat}'] = np.nan
        stats[f'{prefix}pregame_avg_3pa_rate'] = np.nan
        stats[f'{prefix}pregame_ema_3pa_rate'] = np.nan
        stats[f'{prefix}pregame_scoring_margin_sd'] = np.nan
        return stats

    for stat in BART_STATS:
        if stat in history_df.columns:
            series = history_df[stat].astype(float)
            stats[f'{prefix}pregame_avg_{stat}'] = series.mean()
            if n_games > 0:
                stats[f'{prefix}pregame_ema_{stat}'] = series.ewm(alpha=alpha, adjust=False, min_periods=1).mean().iloc[-1]
            else:
                stats[f'{prefix}pregame_ema_{stat}'] = np.nan
        else:
            stats[f'{prefix}pregame_avg_{stat}'] = np.nan
            stats[f'{prefix}pregame_ema_{stat}'] = np.nan

    if '3pa_rate' in history_df.columns:
        series = history_df['3pa_rate'].astype(float)
        stats[f'{prefix}pregame_avg_3pa_rate'] = series.mean()
        if n_games > 0:
            stats[f'{prefix}pregame_ema_3pa_rate'] = series.ewm(alpha=alpha, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            stats[f'{prefix}pregame_ema_3pa_rate'] = np.nan
    else:
        stats[f'{prefix}pregame_avg_3pa_rate'] = np.nan
        stats[f'{prefix}pregame_ema_3pa_rate'] = np.nan

    if 'scoring_margin' in history_df.columns:
        stats[f'{prefix}pregame_scoring_margin_sd'] = history_df['scoring_margin'].astype(float).std()
    else:
        stats[f'{prefix}pregame_scoring_margin_sd'] = np.nan
    
    return stats


# Processes game data chronologically to build cumulative features using team histories
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(SCHOOL_CODE_JSON):
        return
    
    try:
        with open(SCHOOL_CODE_JSON, 'r') as f:
            code_map = json.load(f)
        lower_map = {str(k).lower(): v for k, v in code_map.items()}
    except Exception:
        return

    if not os.path.exists(INPUT_CSV):
        return
    
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        df['date_obj'] = pd.to_datetime(df['date'])
    except Exception:
        return

    df['visitor_opeid'] = df['visitor_team'].apply(
        lambda x: get_opeid(x, code_map, lower_map)
    )
    df['home_opeid'] = df['home_team'].apply(
        lambda x: get_opeid(x, code_map, lower_map)
    )

    shooting_stats = []
    for index, row in df.iterrows():
        box_str = row.get('bart_box_score_array_str')
        vis_fga, vis_3pa = parse_box_score(
            box_str, row['visitor_team'], row['home_team'], code_map, lower_map
        )
        home_fga, home_3pa = parse_box_score(
            box_str, row['home_team'], row['visitor_team'], code_map, lower_map
        )
        shooting_stats.append({
            'index': index,
            'visitor_fga_parsed': vis_fga, 'visitor_3pa_parsed': vis_3pa,
            'home_fga_parsed': home_fga, 'home_3pa_parsed': home_3pa
        })

    shooting_df = pd.DataFrame(shooting_stats).set_index('index')
    df = df.join(shooting_df)

    df['visitor_3pa_rate'] = np.where(df['visitor_fga_parsed'] > 0, 
                                      df['visitor_3pa_parsed'] / df['visitor_fga_parsed'], 0)
    df['home_3pa_rate'] = np.where(df['home_fga_parsed'] > 0, 
                                   df['home_3pa_parsed'] / df['home_fga_parsed'], 0)
    df.drop(columns=['visitor_fga_parsed', 'visitor_3pa_parsed', 
                     'home_fga_parsed', 'home_3pa_parsed'], inplace=True, errors='ignore')

    df.sort_values(by=['season', 'date_obj'], inplace=True)
    
    new_cols_vis = []
    new_cols_home = []
    for stat in BART_STATS:
        new_cols_vis.extend([f'visitor_pregame_avg_{stat}', f'visitor_pregame_ema_{stat}'])
        new_cols_home.extend([f'home_pregame_avg_{stat}', f'home_pregame_ema_{stat}'])

    new_cols_vis.extend(['visitor_pregame_avg_3pa_rate', 'visitor_pregame_ema_3pa_rate', 
                         'visitor_pregame_scoring_margin_sd', 'visitor_games_played_season'])
    new_cols_home.extend(['home_pregame_avg_3pa_rate', 'home_pregame_ema_3pa_rate', 
                          'home_pregame_scoring_margin_sd', 'home_games_played_season'])

    for col in new_cols_vis + new_cols_home:
        df[col] = np.nan

    all_games = []
    for index, game_row in df.iterrows():
        season = game_row['season']
        date = game_row['date_obj']
        visitor_opeid = game_row['visitor_opeid']
        home_opeid = game_row['home_opeid']

        processed_df = pd.DataFrame(all_games)

        # Visitor history
        visitor_history = []
        if not processed_df.empty and pd.notna(visitor_opeid):
            vis_prev_as_vis = processed_df[
                (processed_df['season'] == season) &
                (processed_df['date_obj'] < date) &
                (processed_df['visitor_opeid'] == visitor_opeid)
            ]
            for _, prev_game in vis_prev_as_vis.iterrows():
                stats = {'date_obj': prev_game['date_obj']}
                if pd.notna(prev_game.get('visitor_final_score')) and pd.notna(prev_game.get('home_final_score')):
                    stats['scoring_margin'] = prev_game['visitor_final_score'] - prev_game['home_final_score']
                stats['3pa_rate'] = prev_game.get('visitor_3pa_rate')
                for stat in BART_STATS:
                    stats[stat] = prev_game.get(f'visitor_bart_{stat}_bart')
                visitor_history.append(stats)

            vis_prev_as_home = processed_df[
                (processed_df['season'] == season) &
                (processed_df['date_obj'] < date) &
                (processed_df['home_opeid'] == visitor_opeid)
            ]
            for _, prev_game in vis_prev_as_home.iterrows():
                stats = {'date_obj': prev_game['date_obj']}
                if pd.notna(prev_game.get('home_final_score')) and pd.notna(prev_game.get('visitor_final_score')):
                    stats['scoring_margin'] = prev_game['home_final_score'] - prev_game['visitor_final_score']
                stats['3pa_rate'] = prev_game.get('home_3pa_rate')
                for stat in BART_STATS:
                    stats[stat] = prev_game.get(f'home_bart_{stat}_bart')
                visitor_history.append(stats)
        
        visitor_history_df = pd.DataFrame(visitor_history)
        if not visitor_history_df.empty:
            visitor_history_df.sort_values(by='date_obj', inplace=True)
        
        visitor_stats = calc_pregame_stats(visitor_history_df, "visitor_", EMA_ALPHA)
        for col, val in visitor_stats.items():
            df.loc[index, col] = val
            
        # Home history
        home_history = []
        if not processed_df.empty and pd.notna(home_opeid):
            home_prev_as_vis = processed_df[
                (processed_df['season'] == season) &
                (processed_df['date_obj'] < date) &
                (processed_df['visitor_opeid'] == home_opeid)
            ]
            for _, prev_game in home_prev_as_vis.iterrows():
                stats = {'date_obj': prev_game['date_obj']}
                if pd.notna(prev_game.get('visitor_final_score')) and pd.notna(prev_game.get('home_final_score')):
                    stats['scoring_margin'] = prev_game['visitor_final_score'] - prev_game['home_final_score']
                stats['3pa_rate'] = prev_game.get('visitor_3pa_rate')
                for stat in BART_STATS:
                    stats[stat] = prev_game.get(f'visitor_bart_{stat}_bart')
                home_history.append(stats)

            home_prev_as_home = processed_df[
                (processed_df['season'] == season) &
                (processed_df['date_obj'] < date) &
                (processed_df['home_opeid'] == home_opeid)
            ]
            for _, prev_game in home_prev_as_home.iterrows():
                stats = {'date_obj': prev_game['date_obj']}
                if pd.notna(prev_game.get('home_final_score')) and pd.notna(prev_game.get('visitor_final_score')):
                    stats['scoring_margin'] = prev_game['home_final_score'] - prev_game['visitor_final_score']
                stats['3pa_rate'] = prev_game.get('home_3pa_rate')
                for stat in BART_STATS:
                    stats[stat] = prev_game.get(f'home_bart_{stat}_bart')
                home_history.append(stats)

        home_history_df = pd.DataFrame(home_history)
        if not home_history_df.empty:
            home_history_df.sort_values(by='date_obj', inplace=True)

        home_stats = calc_pregame_stats(home_history_df, "home_", EMA_ALPHA)
        for col, val in home_stats.items():
            df.loc[index, col] = val
        
        all_games.append(game_row.to_dict())

    df.drop(columns=['date_obj'], inplace=True, errors='ignore')

    try:
        df.to_csv(OUTPUT_CSV, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()