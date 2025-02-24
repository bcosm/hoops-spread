# Creates advanced derived features from game statistics and market data
import pandas as pd
import numpy as np

# Generates margin differentials, rolling averages, and market movement features
def calculate_new_features(df_input):
    df_calc = df_input.copy()
    new_features = pd.DataFrame(index=df_calc.index)

    if 'date' not in df_calc.columns:
        return df_input, []
    
    df_calc['date'] = pd.to_datetime(df_calc['date'])

    df_calc['Actual_Margin'] = df_calc['home_final_score'] - df_calc['visitor_final_score']
    df_calc['Visitor_Actual_Margin'] = df_calc['visitor_final_score'] - df_calc['home_final_score']
    new_features['Actual_Margin'] = df_calc['Actual_Margin']
    new_features['Visitor_Actual_Margin'] = df_calc['Visitor_Actual_Margin']

    new_features['Pace_Difference'] = df_calc['home_pregame_ema_tempo_ingame'] - df_calc['visitor_pregame_ema_tempo_ingame']
    new_features['Home_Off_vs_Visitor_Def_Adv'] = df_calc['home_pregame_ema_adj_o_ingame'] - df_calc['visitor_pregame_ema_adj_d_ingame']
    new_features['Visitor_Off_vs_Home_Def_Adv'] = df_calc['visitor_pregame_ema_adj_o_ingame'] - df_calc['home_pregame_ema_adj_d_ingame']
    new_features['Home_eFG_vs_Visitor_Def_eFG'] = df_calc['home_pregame_ema_off_efg_pct_ingame'] - df_calc['visitor_pregame_ema_def_efg_pct_ingame']
    new_features['Visitor_eFG_vs_Home_Def_eFG'] = df_calc['visitor_pregame_ema_off_efg_pct_ingame'] - df_calc['home_pregame_ema_def_efg_pct_ingame']

    spread_movement_values = df_calc['close_home_spread'] - df_calc['open_home_spread']
    new_features['Spread_Movement_Imputed'] = spread_movement_values.isnull().astype(int)
    new_features['Spread_Movement'] = spread_movement_values.fillna(0)

    total_movement_values = df_calc['close_total'] - df_calc['open_total']
    new_features['Total_Movement_Imputed'] = total_movement_values.isnull().astype(int)
    new_features['Total_Movement'] = total_movement_values.fillna(0)

    if 'season' in df_calc.columns and 'home_team' in df_calc.columns and 'visitor_team' in df_calc.columns:
        df_sorted = df_calc.sort_values(by=['season', 'date', 'home_team', 'visitor_team'])
        
        home_rolling_raw = df_sorted.groupby(['season', 'home_team'])['Actual_Margin'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        new_features['Home_Rolling_Margin_Imputed'] = home_rolling_raw.isnull().astype(int)
        new_features['Home_Team_Rolling_Margin_As_Home_Last_3'] = home_rolling_raw.fillna(0)
        
        visitor_rolling_raw = df_sorted.groupby(['season', 'visitor_team'])['Visitor_Actual_Margin'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
        new_features['Visitor_Rolling_Margin_Imputed'] = visitor_rolling_raw.isnull().astype(int)
        new_features['Visitor_Team_Rolling_Margin_As_Visitor_Last_3'] = visitor_rolling_raw.fillna(0)
    else:
        new_features['Home_Team_Rolling_Margin_As_Home_Last_3'] = np.nan
        new_features['Home_Rolling_Margin_Imputed'] = np.nan 
        new_features['Visitor_Team_Rolling_Margin_As_Visitor_Last_3'] = np.nan
        new_features['Visitor_Rolling_Margin_Imputed'] = np.nan

    new_features['Day_of_Week'] = df_calc['date'].dt.dayofweek 
    new_features['Month'] = df_calc['date'].dt.month

    if 'home_bart_opponent_conf_abb' in df_calc.columns and 'visitor_bart_opponent_conf_abb' in df_calc.columns:
        home_conf = df_calc['visitor_bart_opponent_conf_abb'] 
        visitor_conf = df_calc['home_bart_opponent_conf_abb'] 
        new_features['Is_Conference_Game'] = ((home_conf == visitor_conf) &
                                             (home_conf.notna()) &
                                             (visitor_conf.notna())).astype(int)
    else:
        new_features['Is_Conference_Game'] = np.nan

    new_features['Volatility_Diff'] = df_calc['home_pregame_scoring_margin_sd'] - df_calc['visitor_pregame_scoring_margin_sd']
    
    feature_names = new_features.columns.tolist()
    df_augmented = pd.concat([df_input, new_features], axis=1)
    
    return df_augmented, feature_names

# Loads cumulative features and adds advanced derived metrics
def main():
    input_path = "data/features/ncaab_cumulative_features_v11_ref_opeid_ema_cleaned.csv"
    output_path = "data/features/ncaab_cumulative_features_v12_advanced_derived_features.csv"

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    df_augmented, new_feature_names = calculate_new_features(df)

    if df_augmented.equals(df) and not new_feature_names:
        return

    if new_feature_names:
        actual_new_features = [name for name in new_feature_names if name in df_augmented.columns]
        if actual_new_features:
            rows_with_nan = df_augmented[actual_new_features].isnull().any(axis=1).sum()
            if rows_with_nan > 0:
                df_augmented.dropna(subset=actual_new_features, how='any', inplace=True)

    try:
        df_augmented.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == '__main__':
    main()
