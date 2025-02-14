# Assign arena locations to NCAAB teams by scraping Wikipedia and geocoding addresses
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import re
from thefuzz import process as fuzzy_process

SBR_DATA_PATH = os.path.join("data", "raw", "sbr_ncaab_2007_2022.csv")
WIKIPEDIA_ARENAS_URL = "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_basketball_arenas"
OUTPUT_DIR = os.path.join("data", "processed")
FINAL_AUGMENTED_SBR_CSV = os.path.join(OUTPUT_DIR, "sbr_ncaab_with_arena_locations.csv")
UNMATCHED_TEAMS_CSV = os.path.join(OUTPUT_DIR, "sbr_teams_not_in_wikipedia_arenas.csv")

GEOCODER_USER_AGENT = "NCAABettingModelDataCollector/1.5 (your_email@example.com)"
FUZZY_MATCH_THRESHOLD = 85
REQUEST_DELAY_SECONDS = 1.1                                         

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.google.com/'
}

try:
    import lxml
    PARSER = 'lxml'
except ImportError:
    PARSER = 'html.parser'

# Standardize team names for consistent matching between different data sources
def normalize_team_name(name):
    if not isinstance(name, str): 
        return ""
    name = name.lower()
    name = re.sub(r'\bst\.\b', 'saint', name)
    name = re.sub(r'\b(st\.?|saint)\b', 'saint', name)
    name = re.sub(r'\b(state|univ\.?|university)\b', '', name)
    name = re.sub(r'&amp;', 'and', name)
    name = name.replace('&', 'and')
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    replacements = {
        "california": "cal", "southern california": "usc", "louisiana state": "lsu",
        "north carolina": "nc", "uc ": "cal ", "texas christian": "tcu",
        "southern methodist": "smu", "virginia commonwealth": "vcu",
        "florida international": "fiu", "illinois chicago": "uic",
        "texas rio grande valley": "utrgv", "texas el paso": "utep",
        "texas arlington": "ut arlington", "arkansas little rock": "ualr",
        "arkansas pine bluff": "uapb", "siu edwardsville": "southern illinois edwardsville",
        "unc wilmington": "north carolina wilmington", "unc asheville": "north carolina asheville",
        "unc greensboro": "north carolina greensboro", "massachusetts lowell": "umass lowell",
        "bowling green state": "bowling green", "youngstown state": "youngstown st",
        "central connecticut state": "central connecticut", "florida atlantic": "fau",
        "florida gulf coast": "fgcu", "loyola marymount": "lmu", "loyola chicago": "loyola il",
        "mount st marys": "mount saint marys", "saint peters": "saint peter's",
        "saint marys ca": "saint marys college", "md eastern shore": "maryland eastern shore",
        "nj inst of tech": "njit", "cal state": "cs", "csu ": "cs ",
        "fort wayne": "purdue fort wayne", "texas a&m corpus christi": "texas am corpus christi",
        "texas a&m commerce": "texas am commerce", "prairie view a&m": "prairie view am",
        "siu edwardsville": "southern illinois edwardsville", 
        "uc irvine": "cal irvine", "uc riverside": "cal riverside",
        "uc san diego": "cal san diego", "uc santa barbara": "cal santa barbara",
        "uc davis": "cal davis", "massachusetts": "umass", 
        "st thomas mn": "saint thomas", "st bonaventure": "saint bonaventure",
        "st johns": "saint johns", "st francis pa": "saint francis pa",
        "st francis ny": "saint francis ny", "loyola md": "loyola maryland"
    }
    
    for old, new_val in replacements.items():
        if old in name: 
            name = name.replace(old, new_val)
    
    if not any(acronym in name for acronym in ["unc", "utrgv", "uapb", "utep", "siu", "lmu", "fiu", "fau", "fgcu", "a&m", "a-m", "st"]):
         name = name.replace('-', ' ')
    
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Extract arena information for Division I basketball teams from Wikipedia
def scrape_wikipedia_arenas():
    wiki_arenas = []
    try:
        response = requests.get(WIKIPEDIA_ARENAS_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, PARSER)
        tables = soup.find_all('table', class_='wikitable')
        
        if not tables:
            return pd.DataFrame()
        
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue
                
            header_tags = rows[0].find_all(['th', 'td'])
            headers = [th.get_text(strip=True).lower().replace('\n', ' ') for th in header_tags]
            
            header_map = {
                'team': ['team', 'school'], 
                'arena': ['arena'], 
                'city': ['city'], 
                'state': ['state', 'state/province', 'st']
            }
            
            col_indices = {}
            for key, variations in header_map.items():
                for var in variations:
                    try:
                        col_indices[key] = headers.index(var)
                        break
                    except ValueError:
                        continue
            
            if not all(k in col_indices for k in ['team', 'arena', 'city', 'state']):
                continue
            
            for row in rows[1:]:
                cols = row.find_all(['td', 'th'])
                max_needed_idx = max(col_indices.values())
                if len(cols) <= max_needed_idx:
                    continue
                    
                try:
                    school_cell = cols[col_indices['team']]
                    school_link = school_cell.find('a')
                    school_name = school_link.get_text(strip=True) if school_link and school_link.get_text(strip=True) else school_cell.get_text(strip=True)
                    arena_name = cols[col_indices['arena']].get_text(strip=True)
                    city_name = cols[col_indices['city']].get_text(strip=True)
                    state_name = cols[col_indices['state']].get_text(strip=True)
                    
                    school_name, arena_name, city_name, state_name = (
                        re.sub(r'\[.*?\]', '', s).strip() for s in [school_name, arena_name, city_name, state_name]
                    )
                    
                    if school_name and arena_name and city_name and state_name:
                        wiki_arenas.append({
                            'wiki_team_name': school_name, 
                            'normalized_wiki_team': normalize_team_name(school_name),
                            'arena_name': arena_name, 
                            'city': city_name, 
                            'state': state_name
                        })
                except (KeyError, IndexError):
                    continue
                    
    except requests.exceptions.RequestException:
        pass
    except Exception:
        pass
    
    return pd.DataFrame(wiki_arenas) if wiki_arenas else pd.DataFrame()

# Convert city and state information into latitude and longitude coordinates
def geocode_locations(df_to_geocode, geolocator, geocode_function_with_ratelimit):
    if df_to_geocode.empty:
        return df_to_geocode
        
    if not all(col in df_to_geocode.columns for col in ['city', 'state']):
        for col in ['latitude', 'longitude', 'geocoded_address']:
            if col not in df_to_geocode:
                df_to_geocode[col] = None
        return df_to_geocode

    df_to_geocode['address_query'] = df_to_geocode.apply(
        lambda row: f"{row.get('arena_name', row.get('city', ''))}, {row.get('city', '')}, {row.get('state', '')}, USA"
        if pd.notna(row.get('city')) and pd.notna(row.get('state')) else None, axis=1
    )
    
    unique_addresses = df_to_geocode.dropna(subset=['address_query'])['address_query'].unique()
    location_cache = {}
    
    for address in unique_addresses:
        location = None
        try:
            location = geocode_function_with_ratelimit(address, timeout=10)
        except Exception:
            pass
            
        if location:
            location_cache[address] = (location.latitude, location.longitude, location.address)
        else:
            parts = address.split(',')
            if len(parts) >= 3:
                city_state_query = f"{parts[-3].strip()}, {parts[-2].strip()}, USA"
                try:
                    location_city_state = geocode_function_with_ratelimit(city_state_query, timeout=10)
                    if location_city_state:
                        location_cache[address] = (location_city_state.latitude, location_city_state.longitude, location_city_state.address)
                    else:
                        location_cache[address] = (None, None, None)
                except Exception:
                    location_cache[address] = (None, None, None)
            else:
                location_cache[address] = (None, None, None)
    
    df_to_geocode['latitude'] = df_to_geocode['address_query'].map(
        lambda x: location_cache.get(x, (None,None,None))[0] if pd.notna(x) else None
    )
    df_to_geocode['longitude'] = df_to_geocode['address_query'].map(
        lambda x: location_cache.get(x, (None,None,None))[1] if pd.notna(x) else None
    )
    df_to_geocode['geocoded_address'] = df_to_geocode['address_query'].map(
        lambda x: location_cache.get(x, (None,None,None))[2] if pd.notna(x) else None
    )
    
    return df_to_geocode.drop(columns=['address_query'], errors='ignore')

# Main pipeline to match teams with arena locations and geocode them
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SBR_DATA_PATH):
        return
        
    try:
        sbr_df = pd.read_csv(SBR_DATA_PATH)
        sbr_df['visitor_team'] = sbr_df['visitor_team'].astype(str)
        sbr_df['home_team'] = sbr_df['home_team'].astype(str)
        
        visitor_teams = sbr_df['visitor_team'].dropna().unique()
        home_teams = sbr_df['home_team'].dropna().unique()
        sbr_unique_teams_series = pd.Series(list(set(visitor_teams) | set(home_teams))).sort_values()
        
        sbr_teams_map_df = pd.DataFrame({'sbr_team_name': sbr_unique_teams_series})
        sbr_teams_map_df['normalized_sbr_team'] = sbr_teams_map_df['sbr_team_name'].apply(normalize_team_name)
    except Exception:
        return

    wiki_arenas_df = scrape_wikipedia_arenas()
    
    team_locations_list = [] 
    unmatched_sbr_teams_set = set(sbr_teams_map_df['sbr_team_name'])                                    

    if not wiki_arenas_df.empty:
        wiki_normalized_map = {}
        for _, row in wiki_arenas_df.iterrows():
            norm_name = row['normalized_wiki_team']
            if norm_name not in wiki_normalized_map:
                wiki_normalized_map[norm_name] = []
            wiki_normalized_map[norm_name].append(row)
            
        wiki_team_choices = list(wiki_normalized_map.keys())
        
        for _, sbr_row in sbr_teams_map_df.iterrows():
            sbr_team_original = sbr_row['sbr_team_name']
            normalized_sbr = sbr_row['normalized_sbr_team']
            if not normalized_sbr:
                continue

            best_match_tuple = fuzzy_process.extractOne(
                normalized_sbr, wiki_team_choices, score_cutoff=FUZZY_MATCH_THRESHOLD
            )
            
            if best_match_tuple:
                matched_wiki_team_normalized, match_score = best_match_tuple[0], best_match_tuple[1]
                wiki_entries = wiki_normalized_map.get(matched_wiki_team_normalized, [])
                
                if wiki_entries:
                    wiki_entry = wiki_entries[0] 
                    team_locations_list.append({
                        'sbr_team_name': sbr_team_original, 
                        'arena_name': wiki_entry['arena_name'], 
                        'city': wiki_entry['city'],
                        'state': wiki_entry['state']
                    })
                    if sbr_team_original in unmatched_sbr_teams_set:
                        unmatched_sbr_teams_set.remove(sbr_team_original)
    
    if unmatched_sbr_teams_set:
        pd.DataFrame({'sbr_team_name': sorted(list(unmatched_sbr_teams_set))}).to_csv(UNMATCHED_TEAMS_CSV, index=False)

    if not team_locations_list: 
        return

    team_locations_df = pd.DataFrame(team_locations_list)
    team_locations_df.drop_duplicates(subset=['sbr_team_name'], keep='first', inplace=True)

    geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT)
    geocode_rl = RateLimiter(
        geolocator.geocode, 
        min_delay_seconds=REQUEST_DELAY_SECONDS, 
        error_wait_seconds=10.0, 
        max_retries=2, 
        swallow_exceptions=True
    )
    
    geocoded_team_locations_df = geocode_locations(team_locations_df.copy(), geolocator, geocode_rl)

    successfully_geocoded_teams_df = geocoded_team_locations_df.dropna(subset=['latitude', 'longitude'])

    if successfully_geocoded_teams_df.empty:
        pd.DataFrame(columns=sbr_df.columns.tolist() + [
            'visitor_arena', 'visitor_city', 'visitor_state', 'visitor_latitude', 'visitor_longitude', 
            'home_arena', 'home_city', 'home_state', 'home_latitude', 'home_longitude'
        ]).to_csv(FINAL_AUGMENTED_SBR_CSV, index=False)
        return

    sbr_df_augmented = pd.merge(
        sbr_df, 
        successfully_geocoded_teams_df,
        left_on='visitor_team',
        right_on='sbr_team_name',
        how='left',
        suffixes=('', '_visitor_loc')
    )
    sbr_df_augmented.rename(columns={
        'arena_name': 'visitor_arena', 'city': 'visitor_city', 'state': 'visitor_state',
        'latitude': 'visitor_latitude', 'longitude': 'visitor_longitude',
        'geocoded_address': 'visitor_geocoded_address'
    }, inplace=True)
    sbr_df_augmented.drop(columns=['sbr_team_name'], inplace=True, errors='ignore')

    sbr_df_augmented = pd.merge(
        sbr_df_augmented,
        successfully_geocoded_teams_df,
        left_on='home_team',
        right_on='sbr_team_name',
        how='left',
        suffixes=('', '_home_loc')
    )
    sbr_df_augmented.rename(columns={
        'arena_name': 'home_arena', 'city': 'home_city', 'state': 'home_state',
        'latitude': 'home_latitude', 'longitude': 'home_longitude',
        'geocoded_address': 'home_geocoded_address'
    }, inplace=True)
    sbr_df_augmented.drop(columns=['sbr_team_name'], inplace=True, errors='ignore')

    sbr_df_augmented.dropna(subset=['visitor_latitude', 'home_latitude'], inplace=True)

    final_columns = sbr_df.columns.tolist() + [
        'visitor_arena', 'visitor_city', 'visitor_state', 'visitor_latitude', 'visitor_longitude', 'visitor_geocoded_address',
        'home_arena', 'home_city', 'home_state', 'home_latitude', 'home_longitude', 'home_geocoded_address'
    ]
    
    final_columns_existing = [col for col in final_columns if col in sbr_df_augmented.columns]
    sbr_df_final = sbr_df_augmented[final_columns_existing]

    sbr_df_final.to_csv(FINAL_AUGMENTED_SBR_CSV, index=False)

if __name__ == "__main__":
    main()
