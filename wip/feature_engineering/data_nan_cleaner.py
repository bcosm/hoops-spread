# Filters dataset rows based on maximum allowed NaN values per row
import pandas as pd
import numpy as np
import os

INPUT_CSV = os.path.join("data", "features", "ncaab_cumulative_features_v10_no_ml_sd_nans.csv")
OUTPUT_CSV = os.path.join("data", "features", "ncaab_cumulative_features_v11_ref_opeid_ema_cleaned.csv")
MAX_NANS = 3

# Removes rows with more than specified number of NaN values
def filter_rows(input_path, output_path, max_nans):   
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

    nan_counts = df.isna().sum(axis=1)
    df_filtered = df[nan_counts <= max_nans].copy()

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_filtered.to_csv(output_path, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    filter_rows(INPUT_CSV, OUTPUT_CSV, MAX_NANS)