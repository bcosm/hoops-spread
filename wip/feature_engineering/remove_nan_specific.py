# Removes rows with NaN values in specific critical coaching EMA columns
import pandas as pd
import os
import numpy as np

INPUT_CSV = os.path.join("data", "features", "ncaab_cumulative_features_v9_fixed_coach_seasons_debug.csv")
OUTPUT_CSV = os.path.join("data", "features", "ncaab_cumulative_features_v10_no_ml_sd_nans.csv")
COLUMNS_TO_CHECK = [
    'visitor_pg_coach_ema_win_pct_this_season',
    'home_pg_coach_ema_win_pct_this_season',
]

# Filters out rows where any of the specified columns contain NaN values
def remove_nan_rows(input_path, output_path, columns):
    if not os.path.exists(input_path):
        return

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception:
        return

    if df.empty:
        try:
            pd.DataFrame(columns=df.columns).to_csv(output_path, index=False)
        except Exception:
            pass
        return

    df_cleaned = df.dropna(subset=columns, how='any').copy()

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    remove_nan_rows(INPUT_CSV, OUTPUT_CSV, COLUMNS_TO_CHECK)