
# Unit tests for feature engineering functions including rolling stats, matchup features, and derived metrics
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from datetime import datetime, timedelta

import pytest, importlib
if not importlib.util.find_spec("hoops_spread.feature_engineering"):
    pytest.skip("feature_engineering module not implemented yet", allow_module_level=True)


# Test class for individual feature engineering functions with mock data inputs
class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        self.sample_games = pd.DataFrame({
            'date': pd.to_datetime(['2022-11-15', '2022-11-20', '2022-11-25', '2022-12-01']),
            'team': ['Duke', 'Duke', 'Duke', 'Duke'],
            'opponent': ['Virginia', 'UNC', 'Kentucky', 'Louisville'],
            'points_scored': [75, 82, 78, 85],
            'points_allowed': [70, 80, 81, 79],
            'rebounds': [35, 40, 38, 42],
            'assists': [18, 15, 20, 22],
            'turnovers': [12, 15, 10, 8],
            'home_game': [True, False, True, False]
        })
        
        self.sample_team_stats = pd.DataFrame({
            'team': ['Duke', 'UNC', 'Virginia', 'Kentucky'],
            'avg_points': [80.5, 78.2, 72.1, 85.3],
            'avg_rebounds': [38.2, 35.8, 40.1, 36.9],
            'defensive_rating': [95.2, 98.5, 92.8, 100.1]
        })
    
    def test_calculate_rolling_stats(self):
        result = calculate_rolling_stats(
            self.sample_games, 
            columns=['points_scored', 'points_allowed', 'rebounds'],
            window=3,
            groupby='team'
        )
        
        expected_cols = ['points_scored_roll_3', 'points_allowed_roll_3', 'rebounds_roll_3']
        for col in expected_cols:
            self.assertIn(col, result.columns)
        
        self.assertTrue(pd.isna(result['points_scored_roll_3'].iloc[0]))
        self.assertTrue(pd.isna(result['points_scored_roll_3'].iloc[1]))
        
        self.assertIsNotNone(result['points_scored_roll_3'].iloc[2])
    
    def test_create_matchup_features(self):
        team_a_stats = pd.DataFrame({
            'team': ['Duke'],
            'avg_points': [80.5],
            'avg_rebounds': [38.2],
            'defensive_rating': [95.2]
        })
        
        team_b_stats = pd.DataFrame({
            'team': ['UNC'],
            'avg_points': [78.2],
            'avg_rebounds': [35.8],
            'defensive_rating': [98.5]
        })
        
        result = create_matchup_features(team_a_stats, team_b_stats)
        
        self.assertIn('points_differential', result.columns)
        self.assertIn('rebounds_differential', result.columns)
        self.assertIn('defensive_rating_differential', result.columns)
        
        self.assertAlmostEqual(result['points_differential'].iloc[0], 80.5 - 78.2, places=1)
        self.assertAlmostEqual(result['defensive_rating_differential'].iloc[0], 95.2 - 98.5, places=1)
    
    def test_calculate_strength_of_schedule(self):
        schedule = pd.DataFrame({
            'team': ['Duke', 'Duke', 'Duke'],
            'opponent': ['UNC', 'Virginia', 'Kentucky'],
            'date': pd.to_datetime(['2022-11-15', '2022-11-20', '2022-11-25'])
        })
        
        opponent_ratings = pd.DataFrame({
            'team': ['UNC', 'Virginia', 'Kentucky'],
            'rating': [92.5, 88.2, 95.1]
        })
        
        result = calculate_strength_of_schedule(schedule, opponent_ratings)
        
        self.assertIn('sos_rating', result.columns)
        self.assertIn('avg_opponent_rating', result.columns)
        
        expected_sos = (92.5 + 88.2 + 95.1) / 3
        self.assertAlmostEqual(result['avg_opponent_rating'].iloc[0], expected_sos, places=1)
    
    def test_engineer_coaching_features(self):
        games = pd.DataFrame({
            'date': pd.to_datetime(['2022-11-15', '2022-11-20', '2022-11-25']),
            'team': ['Duke', 'Duke', 'Duke'],
            'coach': ['Coach K', 'Coach K', 'Coach K'],
            'season': [2023, 2023, 2023],
            'result': ['W', 'L', 'W']
        })
        
        coaching_history = pd.DataFrame({
            'coach': ['Coach K'],
            'career_wins': [1200],
            'career_losses': [300],
            'years_experience': [40],
            'championships': [5]
        })
        
        result = engineer_coaching_features(games, coaching_history)
        
        self.assertIn('coach_win_pct', result.columns)
        self.assertIn('coach_experience', result.columns)
        self.assertIn('coach_championships', result.columns)
        
        expected_win_pct = 1200 / (1200 + 300)
        self.assertAlmostEqual(result['coach_win_pct'].iloc[0], expected_win_pct, places=3)
    
    def test_create_derived_features(self):
        base_stats = pd.DataFrame({
            'points_scored': [80, 75, 85],
            'points_allowed': [75, 78, 82],
            'field_goals_made': [30, 28, 32],
            'field_goals_attempted': [65, 62, 68],
            'three_pointers_made': [8, 6, 10],
            'three_pointers_attempted': [20, 18, 25],
            'rebounds': [40, 35, 45],
            'turnovers': [12, 15, 10]
        })
        
        result = create_derived_features(base_stats)
        
        self.assertIn('point_differential', result.columns)
        self.assertIn('field_goal_percentage', result.columns)
        self.assertIn('three_point_percentage', result.columns)
        self.assertIn('turnover_rate', result.columns)
        
        self.assertEqual(result['point_differential'].iloc[0], 80 - 75)
        self.assertAlmostEqual(result['field_goal_percentage'].iloc[0], 30/65, places=3)
        self.assertAlmostEqual(result['three_point_percentage'].iloc[0], 8/20, places=3)
    
    def test_calculate_momentum_features(self):
        recent_games = pd.DataFrame({
            'date': pd.to_datetime(['2022-11-10', '2022-11-15', '2022-11-20', '2022-11-25']),
            'team': ['Duke', 'Duke', 'Duke', 'Duke'],
            'result': ['W', 'L', 'W', 'W'],
            'point_differential': [5, -3, 8, 12],
            'home_game': [True, False, True, False]
        })
        
        result = calculate_momentum_features(recent_games, window=3)
        
        self.assertIn('recent_win_pct', result.columns)
        self.assertIn('avg_point_differential', result.columns)
        self.assertIn('momentum_score', result.columns)
        
        self.assertAlmostEqual(result['recent_win_pct'].iloc[-1], 2/3, places=3)


# Integration test class for complete feature engineering pipeline with comprehensive game data
class TestFeatureEngineeringIntegration(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        dates = pd.date_range('2022-11-01', periods=20, freq='5D')
        self.comprehensive_games = pd.DataFrame({
            'date': dates,
            'team': ['Duke'] * 20,
            'opponent': ['UNC', 'Virginia', 'Kentucky'] * 6 + ['Louisville', 'Wake Forest'],
            'points_scored': np.random.randint(70, 90, 20),
            'points_allowed': np.random.randint(65, 85, 20),
            'rebounds': np.random.randint(30, 45, 20),
            'assists': np.random.randint(10, 25, 20),
            'turnovers': np.random.randint(8, 18, 20),
            'field_goals_made': np.random.randint(25, 35, 20),
            'field_goals_attempted': np.random.randint(55, 70, 20),
            'home_game': [True, False] * 10
        })
        
        self.comprehensive_games['result'] = np.where(
            self.comprehensive_games['points_scored'] > self.comprehensive_games['points_allowed'],
            'W', 'L'
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    # Test the complete feature engineering workflow from raw game data to engineered features
    def test_full_feature_engineering_pipeline(self):
        games = self.comprehensive_games.copy()
        
        games = calculate_rolling_stats(
            games,
            columns=['points_scored', 'points_allowed', 'rebounds', 'assists'],
            window=5,
            groupby='team'
        )
        
        games = create_derived_features(games)
        
        games = calculate_momentum_features(games, window=3)
        
        self.assertGreater(len(games.columns), len(self.comprehensive_games.columns))
        
        expected_features = [
            'points_scored_roll_5',
            'point_differential',
            'field_goal_percentage',
            'recent_win_pct',
            'momentum_score'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, games.columns)
        
        self.assertTrue(games['points_scored_roll_5'].iloc[:4].isna().any())
        
        self.assertFalse(games['points_scored_roll_5'].iloc[5:].isna().any())


if __name__ == '__main__':
    unittest.main()
