# Fixes duplicate school names by standardizing them using mapping rules
import pandas as pd
import json
from collections import defaultdict
import os

# Loads school name to code mapping from JSON file
def load_mapping(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Identifies school names that map to the same code creating duplicates
def find_duplicates(mapping):
    code_to_names = defaultdict(list)
    for name, code in mapping.items():
        code_to_names[code].append(name)
    return {code: names for code, names in code_to_names.items() if len(names) > 1}

# Creates replacement mapping choosing longest name as canonical form
def create_replacements(duplicates):
    replacements = {}
    for code, names in duplicates.items():
        longest = max(names, key=len)
        for name in names:
            if name != longest:
                replacements[name] = longest
    return replacements

# Returns manual mapping for state university name standardization
def create_state_mapping():
    return {
        'AlabamaState': 'Alabama St.',
        'AlcornState': 'Alcorn St.',
        'AppalachianState': 'Appalachian St.',
        'ArizonaState': 'Arizona St.',
        'ArkansasState': 'Arkansas St.',
        'BallState': 'Ball St.',
        'BoiseState': 'Boise St.',
        'ChicagoState': 'Chicago St.',
        'ClevelandState': 'Cleveland St.',
        'ColoradoState': 'Colorado St.',
        'CoppinState': 'Coppin St.',
        'DelawareState': 'Delaware St.',
        'EastTennState': 'East Tennessee St.',
        'FloridaState': 'Florida St.',
        'FresnoState': 'Fresno St.',
        'GeorgiaState': 'Georgia St.',
        'IdahoState': 'Idaho St.',
        'IllinoisState': 'Illinois St.',
        'IndianaState': 'Indiana St.',
        'IowaState': 'Iowa St.',
        'JacksonState': 'Jackson St.',
        'JacksonvilleState': 'Jacksonville St.',
        'KansasState': 'Kansas St.',
        'KentState': 'Kent St.',
        'LongBeachState': 'Long Beach St.',
        'McNeeseState': 'McNeese St.',
        'MichiganState': 'Michigan St.',
        'MississippiState': 'Mississippi St.',
        'MissouriState': 'Missouri St.',
        'MontanaState': 'Montana St.',
        'MoreheadState': 'Morehead St.',
        'MorganState': 'Morgan St.',
        'MurrayState': 'Murray St.',
        'NCState': 'N.C. State',
        'NewMexicoState': 'New Mexico St.',
        'NichollsState': 'Nicholls St.',
        'NorfolkState': 'Norfolk St.',
        'NorthDakotaState': 'North Dakota St.',
        'NorthwesternState': 'Northwestern St.',
        'OhioState': 'Ohio St.',
        'OklahomaState': 'Oklahoma St.',
        'OregonState': 'Oregon St.',
        'PennState': 'Penn St.',
        'PortlandState': 'Portland St.',
        'SacramentoState': 'Sacramento St.',
        'SamHoustonState': 'Sam Houston St.',
        'SanDiegoState': 'San Diego St.',
        'SanJoseState': 'San Jose St.',
        'SavannahState': 'Savannah St.',
        'SouthCarolinaState': 'South Carolina St.',
        'SouthDakotaState': 'South Dakota St.',
        'SEMissouriState': 'Southeast Missouri St.',
        'TarletonState': 'Tarleton St.',
        'TennesseeState': 'Tennessee St.',
        'TexasState': 'Texas St.',
        'TowsonState': 'Towson St.',
        'UtahState': 'Utah St.',
        'WashingtonState': 'Washington St.',
        'WeberState': 'Weber St.',
        'WichitaState': 'Wichita St.',
        'WinstonSalemState': 'Winston Salem St.',
        'WrightState': 'Wright St.',
        'YoungstownState': 'Youngstown St.',
    }

# Processes CSV in chunks applying name replacements to all team columns
def process_csv(csv_path, replacements, output_path, chunk_size=10000):
    sample_df = pd.read_csv(csv_path, nrows=1)
    columns = sample_df.columns.tolist()
    
    team_columns = [col for col in columns if any(keyword in col.lower() for keyword in ['team', 'visitor', 'home'])]
    
    first_chunk = True
    total_replacements = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_replacements = 0
        for col in team_columns:
            if col in chunk.columns:
                before = chunk[col].value_counts()
                chunk[col] = chunk[col].replace(replacements)
                for old_name in replacements:
                    if old_name in before:
                        chunk_replacements += before.get(old_name, 0)
        
        total_replacements += chunk_replacements
        
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    return total_replacements

# Orchestrates duplicate school name fixing using mapping and state standardization
def main():
    json_path = os.path.join("data", "processed_with_school_codes", "school_name_to_code.json")
    csv_path = os.path.join("data", "features", "ncaab_cumulative_features_v12_advanced_derived_features.csv")
    output_path = os.path.join("data", "features", "ncaab_cumulative_features_v12_advanced_derived_features_fixed.csv")
    
    if not os.path.exists(json_path) or not os.path.exists(csv_path):
        return
    
    try:
        mapping = load_mapping(json_path)
        duplicates = find_duplicates(mapping)
        replacements = {}
        
        if duplicates:
            replacements.update(create_replacements(duplicates))
        
        state_mapping = create_state_mapping()
        replacements.update(state_mapping)
        
        if replacements:
            process_csv(csv_path, replacements, output_path)
    except Exception:
        pass

if __name__ == "__main__":
    main()
