# Fetch college basketball schedule and odds data from Sportsbook Review Online
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import io

BASE_ARCHIVE_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaa-basketball-20{start_year_short}-{end_year_short}/"
BASE_EXCEL_URL = "https://www.sportsbookreviewsonline.com/wp-content/uploads/sportsbookreviewsonline_com_737/"
EXCEL_FILENAME_PATTERN = "ncaa-basketball-{start_year_full}-{end_year_short}.xlsx"

OLDEST_SEASON_END_YEAR = 2008
NEWEST_SEASON_END_YEAR = 2022
OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = "data/raw/sbr_schedules.csv"
REQUEST_DELAY_SECONDS = 2.5

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    import lxml
    PARSER = 'lxml'
except ImportError:
    PARSER = 'html.parser'

def parse_odds_pair(val1_str, val2_str):
    try:
        val1 = float(val1_str)
        val2 = float(val2_str)
        if abs(val1) < abs(val2) and abs(val1) < 50:
            if val2 > 100: return val1, val2, True
            else: return val1, val2, True 
        elif abs(val2) < abs(val1) and abs(val2) < 50:
            if val1 > 100: return val2, val1, False
            else: return val2, val1, False
        elif val1 > 100 and abs(val2) < 50 : return val2, val1, False
        elif val2 > 100 and abs(val1) < 50 : return val1, val2, True
        if abs(val1) < abs(val2): return val1, val2, True
        else: return val2, val1, False
    except (ValueError, TypeError):
        return None, None, None

def assign_year_to_date(date_str, start_year_of_season, end_year_of_season):
    if not date_str: return None
    date_str_normalized = str(date_str).replace('/', '').zfill(4) if isinstance(date_str, (int, float)) else str(date_str).replace('/', '')
    
    if isinstance(date_str, (float, int)) and date_str > 30000 and date_str < 60000:
        try:
            dt_obj = pd.to_datetime('1899-12-30') + pd.to_timedelta(date_str, unit='D')
            month = dt_obj.month
            year_to_assign = start_year_of_season if 8 <= month <= 12 else end_year_of_season
            if dt_obj.year == year_to_assign:
                return dt_obj.strftime('%Y-%m-%d')
            else:
                date_str_normalized = dt_obj.strftime('%m%d') 
        except:
            date_str_normalized = ''.join(filter(str.isdigit, str(date_str)))
    else:
        date_str_normalized = ''.join(filter(str.isdigit, str(date_str)))

    if len(date_str_normalized) == 3: date_str_normalized = "0" + date_str_normalized
    if len(date_str_normalized) != 4:
        return None
    try:
        month = int(date_str_normalized[:2])
        day = int(date_str_normalized[2:])
        year_to_assign = start_year_of_season if 8 <= month <= 12 else end_year_of_season
        return datetime(year_to_assign, month, day).strftime('%Y-%m-%d')
    except ValueError:
        return None

def process_game_rows(row1_data, row2_data, season_label, season_start_year, season_end_year):
    game_details = {}
    try:
        def get_cell_text(cell_data, index):
            if isinstance(cell_data, pd.Series):
                return str(cell_data.iloc[index]).strip() if index < len(cell_data) and pd.notna(cell_data.iloc[index]) else ""
            else:
                return cell_data[index].get_text(strip=True) if index < len(cell_data) else ""

        min_cols = 11 
        if (isinstance(row1_data, pd.Series) and len(row1_data) < min_cols) or\
           (not isinstance(row1_data, pd.Series) and len(row1_data) < min_cols):
            return None
        if (isinstance(row2_data, pd.Series) and len(row2_data) < min_cols) or\
           (not isinstance(row2_data, pd.Series) and len(row2_data) < min_cols):
            return None

        vh1 = get_cell_text(row1_data, 2).upper()
        vh2 = get_cell_text(row2_data, 2).upper()

        if vh1 == 'V' and vh2 == 'H':
            visitor_data, home_data = row1_data, row2_data
        elif vh1 == 'H' and vh2 == 'V':
            visitor_data, home_data = row2_data, row1_data
        else:
            return None

        raw_date = get_cell_text(visitor_data, 0)
        game_date = assign_year_to_date(raw_date, season_start_year, season_end_year)
        if not game_date: return None

        game_details = {
            'date': game_date, 'season': season_label,
            'visitor_rot': get_cell_text(visitor_data, 1),
            'visitor_team': get_cell_text(visitor_data, 3),
            'visitor_1st_score': pd.to_numeric(get_cell_text(visitor_data, 4), errors='coerce'),
            'visitor_2nd_score': pd.to_numeric(get_cell_text(visitor_data, 5), errors='coerce'),
            'visitor_final_score': pd.to_numeric(get_cell_text(visitor_data, 6), errors='coerce'),
            'visitor_ml': get_cell_text(visitor_data, 9),
            'home_rot': get_cell_text(home_data, 1),
            'home_team': get_cell_text(home_data, 3),
            'home_1st_score': pd.to_numeric(get_cell_text(home_data, 4), errors='coerce'),
            'home_2nd_score': pd.to_numeric(get_cell_text(home_data, 5), errors='coerce'),
            'home_final_score': pd.to_numeric(get_cell_text(home_data, 6), errors='coerce'),
            'home_ml': get_cell_text(home_data, 9),
        }

        for odds_type_key, cell_idx in [('open', 7), ('close', 8), ('2h', 10)]:
            val_v_str = get_cell_text(visitor_data, cell_idx)
            val_h_str = get_cell_text(home_data, cell_idx)
            spread, total, spread_on_visitor = parse_odds_pair(val_v_str, val_h_str)
            game_details[f'{odds_type_key}_total'] = total
            if spread_on_visitor is not None:
                game_details[f'{odds_type_key}_visitor_spread'] = spread if spread_on_visitor else (-spread if spread is not None else None)
                game_details[f'{odds_type_key}_home_spread'] = (-spread if spread is not None else None) if spread_on_visitor else spread
            else:
                game_details[f'{odds_type_key}_visitor_spread'], game_details[f'{odds_type_key}_home_spread'] = None, None
        return game_details
    except Exception as e:
        return None

def fetch_data_for_url(url, season_label, season_start_year, season_end_year):
    processed_games = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=45)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()

        if 'excel' in content_type or 'spreadsheetml' in content_type or url.endswith('.xlsx') or url.endswith('.xls'):
            excel_data = pd.read_excel(io.BytesIO(response.content), header=0)
            if excel_data.empty or len(excel_data.columns) < 11:
                return []
            if len(excel_data) % 2 != 0:
                excel_data = excel_data[:-1]
            for i in range(0, len(excel_data) -1, 2):
                game = process_game_rows(excel_data.iloc[i], excel_data.iloc[i+1], season_label, season_start_year, season_end_year)
                if game: processed_games.append(game)
        elif 'html' in content_type:
            soup = BeautifulSoup(response.content, PARSER)
            table = soup.find('table', class_="table-sm")
            if not table: table = soup.find('table', class_="table-bordered")
            if not table:
                return []
            tbody = table.find('tbody')
            if not tbody:
                return []
            rows = tbody.find_all('tr')
            game_rows = rows[1:]
            if len(game_rows) % 2 != 0:
                game_rows = game_rows[:-1]
            for i in range(0, len(game_rows) - 1, 2):
                game = process_game_rows(game_rows[i].find_all('td'), game_rows[i+1].find_all('td'), season_label, season_start_year, season_end_year)
                if game: processed_games.append(game)
        else:
            return []
        return processed_games
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404:
            return None
        else:
            return None
    except Exception as e:
        return None                                  

def main():
    all_seasons_data = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for current_season_end_year in range(NEWEST_SEASON_END_YEAR, OLDEST_SEASON_END_YEAR - 1, -1):
        current_season_start_year = current_season_end_year - 1
        start_year_short = str(current_season_start_year)[-2:]
        end_year_short = str(current_season_end_year)[-2:]
        season_label_str = f"{current_season_start_year}-{current_season_end_year}"

        archive_url = BASE_ARCHIVE_URL.format(start_year_short=start_year_short, end_year_short=end_year_short)
        season_data = fetch_data_for_url(archive_url, season_label_str, current_season_start_year, current_season_end_year)

        if season_data is None or not season_data:
            excel_filename = EXCEL_FILENAME_PATTERN.format(start_year_full=current_season_start_year, end_year_short=end_year_short)
            excel_url = BASE_EXCEL_URL + excel_filename
            season_data = fetch_data_for_url(excel_url, season_label_str, current_season_start_year, current_season_end_year)

        if season_data:
            all_seasons_data.extend(season_data)

        if current_season_end_year > OLDEST_SEASON_END_YEAR:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not all_seasons_data:
        return

    df = pd.DataFrame(all_seasons_data)

    for col in ['visitor_ml', 'home_ml']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    score_cols = ['visitor_1st_score', 'visitor_2nd_score', 'visitor_final_score',
                  'home_1st_score', 'home_2nd_score', 'home_final_score']
    for col in score_cols:
        if col in df.columns: df[col] = df[col].astype('Int64')
    odds_value_cols = [col for col in df.columns if 'spread' in col or 'total' in col]
    for col in odds_value_cols:
         if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    column_order = [
        'date', 'season',
        'visitor_rot', 'visitor_team', 'visitor_1st_score', 'visitor_2nd_score', 'visitor_final_score', 'visitor_ml',
        'home_rot', 'home_team', 'home_1st_score', 'home_2nd_score', 'home_final_score', 'home_ml',
        'open_visitor_spread', 'open_home_spread', 'open_total',
        'close_visitor_spread', 'close_home_spread', 'close_total',
        '2h_visitor_spread', '2h_home_spread', '2h_total'
    ]
    existing_cols = [col for col in column_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + remaining_cols]

    try:
        df.to_csv(OUTPUT_FILE, index=False)
    except IOError as e:
        pass

if __name__ == "__main__":
    main()
