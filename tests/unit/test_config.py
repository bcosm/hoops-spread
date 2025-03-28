
# Test configuration and data generation utilities for unit testing

import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

TEST_DATA_DIR = Path(__file__).parent / "test_data"

TEST_CONFIGS = {
    'data_collection': {
        'oldest_season_end_year': 2020,
        'newest_season_end_year': 2021,
        'request_delay_seconds': 0.1,
        'max_retries': 2
    },
    'preprocessing': {
        'missing_value_threshold': 0.5,
        'duplicate_threshold': 0.95,
        'min_games_per_team': 10
    },
    'feature_engineering': {
        'rolling_window': 5,
        'momentum_window': 3,
        'sos_min_games': 5
    },
    'modeling': {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 3,
        'max_evals': 10
    }
}

# Generate synthetic basketball game data for testing
def create_test_game_data(n_games: int = 100, n_teams: int = 10) -> pd.DataFrame:
    np.random.seed(42)
    
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    
    games = []
    start_date = pd.Timestamp('2022-11-01')
    
    for i in range(n_games):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        home_score = np.random.randint(60, 95)
        away_score = np.random.randint(60, 95)
        
        if 'Team_00' in [home_team, away_team]:
            if home_team == 'Team_00':
                home_score += 10
            else:
                away_score += 10
        
        game = {
            'date': start_date + pd.Timedelta(days=i // 5),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_rebounds': np.random.randint(25, 45),
            'away_rebounds': np.random.randint(25, 45),
            'home_assists': np.random.randint(10, 25),
            'away_assists': np.random.randint(10, 25),
            'home_turnovers': np.random.randint(8, 18),
            'away_turnovers': np.random.randint(8, 18),
            'home_fouls': np.random.randint(12, 22),
            'away_fouls': np.random.randint(12, 22)
        }
        
        games.append(game)
    
    return pd.DataFrame(games)

# Generate synthetic team metadata for testing
def create_test_team_data(teams: list) -> pd.DataFrame:
    np.random.seed(42)
    
    team_data = []
    for team in teams:
        data = {
            'team': team,
            'conference': np.random.choice(['ACC', 'SEC', 'Big Ten', 'Big 12', 'Pac-12']),
            'ranking': np.random.randint(1, 351) if np.random.random() > 0.7 else None,
            'kenpom_rating': np.random.normal(0, 10),
            'coach': f"Coach_{team.split('_')[1]}",
            'arena_capacity': np.random.randint(5000, 25000),
            'altitude': np.random.randint(0, 5000)
        }
        team_data.append(data)
    
    return pd.DataFrame(team_data)

# Generate synthetic sentiment data for testing
def create_test_sentiment_data(teams: list, n_days: int = 30) -> pd.DataFrame:
    np.random.seed(42)
    
    sentiment_data = []
    start_date = pd.Timestamp('2022-11-01')
    
    for team in teams:
        for day in range(n_days):
            date = start_date + pd.Timedelta(days=day)
            data = {
                'date': date,
                'team': team,
                'sentiment_score': np.random.normal(0, 0.3),
                'mention_count': np.random.poisson(10),
                'positive_mentions': np.random.poisson(6),
                'negative_mentions': np.random.poisson(4),
                'compound_sentiment': np.random.normal(0, 0.5)
            }
            sentiment_data.append(data)
    
    return pd.DataFrame(sentiment_data)

# Manager for temporary files and directories used in tests
class TestDataManager:
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
    
    def create_temp_dir(self) -> str:
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, suffix: str = '.csv') -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self.temp_files.append(temp_file.name)
        temp_file.close()
        return temp_file.name
    
    def cleanup(self):
        import shutil
        
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass
        
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except (OSError, FileNotFoundError):
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()

# Validate dataframe structure matches expected schema
def assert_dataframe_structure(df: pd.DataFrame, expected_columns: list, 
                             min_rows: int = 1) -> None:
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
    
    missing_columns = set(expected_columns) - set(df.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"

def assert_no_infinite_values(df: pd.DataFrame, columns: Optional[list] = None) -> None:
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        infinite_count = np.isinf(df[col]).sum()
        assert infinite_count == 0, f"Column '{col}' contains {infinite_count} infinite values"

def assert_no_duplicate_games(df: pd.DataFrame, 
                            key_columns: list = ['date', 'home_team', 'away_team']) -> None:
    duplicate_count = df.duplicated(subset=key_columns).sum()
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate games"

SAMPLE_GAME_DATA = create_test_game_data(50, 8)
SAMPLE_TEAM_DATA = create_test_team_data(SAMPLE_GAME_DATA['home_team'].unique())
SAMPLE_SENTIMENT_DATA = create_test_sentiment_data(SAMPLE_TEAM_DATA['team'].tolist())
