# Verifies that team name cleanup was applied correctly by comparing before and after
import pandas as pd
import json
import os

# Checks if expected team names were removed and replaced according to mapping
def verify_cleanup():
    original_file = os.path.join("data", "features", "ncaab_cumulative_features_v12_advanced_derived_features.csv")
    cleaned_file = os.path.join("data", "features", "ncaab_cumulative_features_v12_team_names_cleaned.csv")
    mapping_file = "team_name_cleanup_mapping.json"
    
    if not all(os.path.exists(f) for f in [original_file, cleaned_file, mapping_file]):
        return
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        original_sample = pd.read_csv(original_file, nrows=10000)
        cleaned_sample = pd.read_csv(cleaned_file, nrows=10000)
        
        orig_teams = set(original_sample['visitor_team'].dropna().astype(str)).union(
                        set(original_sample['home_team'].dropna().astype(str)))
        cleaned_teams = set(cleaned_sample['visitor_team'].dropna().astype(str)).union(
                          set(cleaned_sample['home_team'].dropna().astype(str)))
        
        removed_teams = orig_teams - cleaned_teams
        expected_removals = set(mapping.keys()).intersection(orig_teams)
        
        return removed_teams == expected_removals
    except Exception:
        return False

if __name__ == "__main__":
    verify_cleanup()
