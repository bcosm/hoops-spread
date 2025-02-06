# Clean college basketball data by removing missing values and unnecessary columns
import os
import pandas as pd
import numpy as np

INPUT_CSV_PATH = os.path.join("data", "processed", "sbr_elo_bart_with_extra_cols_v4.csv")
OUTPUT_DIR = os.path.join("data", "processed") 
CLEANED_CSV_PATH = os.path.join(OUTPUT_DIR, "ncaab_data_cleaned.csv")

# Analyze missing values across all columns to understand data completeness
def analyze_emptiness(df, df_name="DataFrame"):
    empty_stats = []
    for col in df.columns:
        missing_values = df[col].isnull().sum()
        total_values = len(df)
        percentage_empty = (missing_values / total_values) * 100
        empty_stats.append({
            "column": col,
            "missing_values": missing_values,
            "total_values": total_values,
            "percentage_empty": percentage_empty
        })
    
    summary = {
        "total_columns": len(df.columns),
        "100_percent_empty": sum(1 for s in empty_stats if s['percentage_empty'] == 100),
        "some_missing": sum(1 for s in empty_stats if 0 < s['percentage_empty'] < 100),
        "no_missing": sum(1 for s in empty_stats if s['percentage_empty'] == 0)
    }
    return empty_stats

# Main function to load data, perform cleaning operations, and save cleaned dataset
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV_PATH):
        return
    
    try:
        df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
    except Exception:
        return

    rows_before_na_drop = len(df)
    
    df.dropna(subset=['visitor_final_score', 'home_final_score'], how='any', inplace=True)
    rows_before_na_drop = len(df)

    df.dropna(subset=['close_visitor_spread'], inplace=True)

    columns_to_drop = [
        '2h_visitor_spread', '2h_home_spread', '2h_total',
        'visitor_original_sbr_school_name', 'home_original_sbr_school_name',
        'bart_row_id_or_flag'
    ]
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns_to_drop, inplace=True, errors='ignore')

    analyze_emptiness(df, "Cleaned DataFrame")

    try:
        df.to_csv(CLEANED_CSV_PATH, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()
