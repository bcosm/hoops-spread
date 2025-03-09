# Standardizes team names in feature dataset using predefined mapping rules
import pandas as pd
import json
import os

# Extracts all unique team names from visitor and home team columns
def identify_team_names(csv_path, chunk_size=10000):
    all_teams = set()
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if 'visitor_team' in chunk.columns:
            visitor_teams = chunk['visitor_team'].dropna().astype(str)
            all_teams.update(visitor_teams.unique())
        
        if 'home_team' in chunk.columns:
            home_teams = chunk['home_team'].dropna().astype(str)
            all_teams.update(home_teams.unique())
    
    return all_teams

# Creates mapping dictionary for team name standardization using manual rules
def create_mapping(all_teams):
    teams_list = sorted(list(all_teams))
    mapping = {}
    
    manual_mappings = {
        'AlabamaState': 'Alabama St.',
        'ArkansasState': 'Arkansas St.',
        'ArizonaState': 'Arizona St.',
        'BallState': 'Ball St.',
        'BoiseState': 'Boise St.',
        'BostonU': 'Boston University',
        'BowlingGreen': 'Bowling Green',
        'CalSantaBarb': 'Cal St. Santa Barbara',
        'CalSantaBarbara': 'Cal St. Santa Barbara',
        'CentralArkansas': 'Central Arkansas',
        'CentralMichigan': 'Central Michigan',
        'ChicagoSt': 'Chicago St.',
        'ClevelandState': 'Cleveland St.',
        'CoastalCarolina': 'Coastal Carolina',
        'ColoradoState': 'Colorado St.',
        'CoppinState': 'Coppin St.',
        'DenverU': 'Denver',
        'EastCarolina': 'East Carolina',
        'EasternIllinois': 'Eastern Illinois',
        'EasternKentucky': 'Eastern Kentucky',
        'EasternMichigan': 'Eastern Michigan',
        'FloridaState': 'Florida St.',
        'FresnoState': 'Fresno St.',
        'GeorgeMason': 'George Mason',
        'GeorgiaSouthern': 'Georgia Southern',
        'GeorgiaTech': 'Georgia Tech',
        'GrandCanyon': 'Grand Canyon',
        'HighPoint': 'High Point',
        'HolyCross': 'Holy Cross',
        'HoustonU': 'Houston',
        'IdahoState': 'Idaho St.',
        'IllinoisState': 'Illinois St.',
        'IncarnateWord': 'Incarnate Word',
        'IndianaState': 'Indiana St.',
        'IndianaU': 'Indiana',
        'IowaState': 'Iowa St.',
        'JacksonState': 'Jackson St.',
        'JacksonvilleSt': 'Jacksonville St.',
        'JamesMadison': 'James Madison',
        'KansasState': 'Kansas St.',
        'KennesawSt': 'Kennesaw St.',
        'KentState': 'Kent St.',
        'LaSalle': 'La Salle',
        'LouisianaTech': 'Louisiana Tech',
        'LoyolaMaryland': 'Loyola Maryland',
        'McNeeseState': 'McNeese St.',
        'MemphisU': 'Memphis',
        'MichiganState': 'Michigan St.',
        'MinnesotaU': 'Minnesota',
        'MississippiSt': 'Mississippi St.',
        'MissouriState': 'Missouri St.',
        'MontanaState': 'Montana St.',
        'MoreheadState': 'Morehead St.',
        'MorganState': 'Morgan St.',
        'MurrayState': 'Murray St.',
        'NCCharlotte': 'UNC Charlotte',
        'NebraskaOmaha': 'Nebraska Omaha',
        'NewHampshire': 'New Hampshire',
        'NewMexicoState': 'New Mexico St.',
        'NichollsState': 'Nicholls St.',
        'NorfolkSt': 'Norfolk St.',
        'NorthAlabama': 'North Alabama',
        'NorthCarolina': 'North Carolina',
        'NorthDakotaState': 'North Dakota St.',
        'NorthTexas': 'North Texas',
        'NorthernArizona': 'Northern Arizona',
        'NorthernColorado': 'Northern Colorado',
        'NorthernIowa': 'Northern Iowa',
        'NorthernKentucky': 'Northern Kentucky',
        'NorthwesternSt': 'Northwestern St.',
        'NotreDame': 'Notre Dame',
        'OhioState': 'Ohio St.',
        'OklahomaState': 'Oklahoma St.',
        'OldDominion': 'Old Dominion',
        'OralRoberts': 'Oral Roberts',
        'OregonState': 'Oregon St.',
        'PennState': 'Penn St.',
        'PortlandState': 'Portland St.',
        'PrairieViewA&M': 'Prairie View A&M',
        'SacramentoState': 'Sacramento St.',
        'SacredHeart': 'Sacred Heart',
        'SaintLouis': 'Saint Louis',
        'SamHoustonSt': 'Sam Houston St.',
        'SanDiegoState': 'San Diego St.',
        'SantaClara': 'Santa Clara',
        'SeattleU': 'Seattle',
        'SouthAlabama': 'South Alabama',
        'SouthCarolina': 'South Carolina',
        'SouthDakotaState': 'South Dakota St.',
        'SouthFlorida': 'South Florida',
        'SouthernIllinois': 'Southern Illinois',
        'SouthernMiss': 'Southern Miss',
        'SouthernUtah': 'Southern Utah',
        'StephenF.Austin': 'Stephen F. Austin',
        'StonyBrook': 'Stony Brook',
        'TarletonSt': 'Tarleton St.',
        'TennesseeState': 'Tennessee St.',
        'TennesseeTech': 'Tennessee Tech',
        'TexasSouthern': 'Texas Southern',
        'TexasTech': 'Texas Tech',
        'TowsonState': 'Towson',
        'ULLafayette': 'UL Lafayette',
        'UTArlington': 'UT Arlington',
        'UtahState': 'Utah St.',
        'UtahValley': 'Utah Valley',
        'VirginiaTech': 'Virginia Tech',
        'WakeForest': 'Wake Forest',
        'WeberState': 'Weber St.',
        'WestVirginia': 'West Virginia',
        'WesternCarolina': 'Western Carolina',
        'WesternIllinois': 'Western Illinois',
        'WesternKentucky': 'Western Kentucky',
        'WesternMichigan': 'Western Michigan',
        'WichitaState': 'Wichita St.',
        'WrightState': 'Wright St.',
        'YoungstownState': 'Youngstown St.',
    }
    
    for old_name, new_name in manual_mappings.items():
        if old_name in all_teams:
            mapping[old_name] = new_name
    
    return mapping

# Applies team name mapping to CSV data and saves cleaned version
def clean_csv(csv_path, mapping, output_path, chunk_size=10000):
    total_replacements = 0
    first_chunk = True
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_replacements = 0
        
        if 'visitor_team' in chunk.columns:
            before_count = chunk['visitor_team'].value_counts()
            chunk['visitor_team'] = chunk['visitor_team'].replace(mapping)
            
            for old_name in mapping.keys():
                if old_name in before_count:
                    chunk_replacements += before_count.get(old_name, 0)
        
        if 'home_team' in chunk.columns:
            before_count = chunk['home_team'].value_counts()
            chunk['home_team'] = chunk['home_team'].replace(mapping)
            
            for old_name in mapping.keys():
                if old_name in before_count:
                    chunk_replacements += before_count.get(old_name, 0)
        
        total_replacements += chunk_replacements
        
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    return total_replacements

# Orchestrates team name cleanup process and saves mapping to JSON file
def main():
    base_path = r"c:\Users\bcosm\Documents\ncaa_basketball_point_spread"
    csv_path = os.path.join(base_path, "data", "features", "ncaab_cumulative_features_v12_advanced_derived_features.csv")
    output_path = os.path.join(base_path, "data", "features", "ncaab_cumulative_features_v12_team_names_cleaned.csv")
    mapping_path = os.path.join(base_path, "team_name_cleanup_mapping.json")
    
    all_teams = identify_team_names(csv_path)
    mapping = create_mapping(all_teams)
    
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    if mapping:
        total_replacements = clean_csv(csv_path, mapping, output_path)

if __name__ == "__main__":
    main()
