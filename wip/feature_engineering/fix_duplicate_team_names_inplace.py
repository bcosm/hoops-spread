# Identifies and fixes duplicate team names that map to the same school code
import pandas as pd
import json
from collections import defaultdict
import os

# Finds team names that share the same school code and creates replacement mapping
def find_duplicates():
    mapping_file = os.path.join('data', 'processed_with_school_codes', 'school_name_to_code.json')
    csv_file = os.path.join('data', 'features', 'ncaab_cumulative_features_v12_advanced_derived_features.csv')
    
    if not os.path.exists(mapping_file) or not os.path.exists(csv_file):
        return {}, csv_file
    
    try:
        with open(mapping_file, 'r') as f:
            school_to_code = json.load(f)
        
        df = pd.read_csv(csv_file, usecols=['visitor_team', 'home_team'])
    except Exception:
        return {}, csv_file
    
    visitor_teams = set(df['visitor_team'].dropna())
    home_teams = set(df['home_team'].dropna()) 
    all_teams = visitor_teams.union(home_teams)
    
    csv_teams_in_mapping = {team for team in all_teams if team in school_to_code}
    
    code_to_teams = defaultdict(list)
    for team in csv_teams_in_mapping:
        code = school_to_code[team]
        code_to_teams[code].append(team)
    
    replacements = {}
    for code, teams in code_to_teams.items():
        if len(teams) > 1:
            longest = max(teams, key=len)
            for team in teams:
                if team != longest:
                    replacements[team] = longest
    
    return replacements, csv_file

# Applies team name replacements to visitor and home team columns
def apply_replacements(replacements, csv_file):
    if not replacements or not os.path.exists(csv_file):
        return
    
    try:
        chunk_size = 10000
        chunks = []
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            for old_name, new_name in replacements.items():
                chunk.loc[chunk['visitor_team'] == old_name, 'visitor_team'] = new_name
                chunk.loc[chunk['home_team'] == old_name, 'home_team'] = new_name
            chunks.append(chunk)
        
        result_df = pd.concat(chunks, ignore_index=True)
        output_file = csv_file.replace('.csv', '_fixed_team_names.csv')
        result_df.to_csv(output_file, index=False)
    except Exception:
        pass

# Orchestrates duplicate team name detection and fixing process
def main():
    replacements, csv_file = find_duplicates()
    if replacements:
        apply_replacements(replacements, csv_file)

if __name__ == "__main__":
    main()
