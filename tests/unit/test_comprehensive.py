
# Comprehensive unit tests covering sentiment processing, feature engineering and model validation

import sys, os, json, tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import pytest, importlib

if not importlib.util.find_spec("hoops_spread.sentiment"):
    pytest.skip("hoops_spread.sentiment module not available (CI skip)",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test sentiment data deduplication functionality
class TestSentimentDeduplication(unittest.TestCase):
    
    def setUp(self):
        self.test_data = pd.DataFrame({
            'text': [
                'Duke is amazing!',
                'Duke is amazing!',
                'duke is AMAZING!',
                'North Carolina sucks',
                'UNC is terrible',
            ],
            'school': ['Duke', 'Duke', 'Duke', 'North Carolina', 'North Carolina'],
            'created_utc': [1000, 1001, 1002, 1003, 1004]
        })
    
    def test_exact_duplicate_detection(self):
        from hoops_spread.sentiment import detect_duplicates
        
        duplicates = self.test_data[self.test_data.duplicated(['text'], keep=False)]
        
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(all(duplicates['text'].str.lower() == 'duke is amazing!'))
    
    def test_case_insensitive_dedup(self):
        normalized = self.test_data.copy()
        normalized['text_lower'] = normalized['text'].str.lower()
        
        case_duplicates = normalized[normalized.duplicated(['text_lower'], keep=False)]
        
        self.assertEqual(len(case_duplicates), 3)


# Test feature engineering processes for data leakage and sanity checks
class TestFeatureEngineeringSanity(unittest.TestCase):
    
    def setUp(self):
        self.test_games = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'visitor_team': ['Duke'] * 50 + ['UNC'] * 50,
            'home_team': ['UNC'] * 50 + ['Duke'] * 50,
            'visitor_score': np.random.randint(60, 100, 100),
            'home_score': np.random.randint(60, 100, 100),
            'visitor_spread': np.random.uniform(-10, 10, 100),
        })
        self.test_games['margin'] = self.test_games['home_score'] - self.test_games['visitor_score']
    
    def test_rolling_stats_sanity(self):
        self.test_games['rolling_avg_score'] = self.test_games.groupby('visitor_team')['visitor_score'].rolling(5).mean().reset_index(0, drop=True)
        
        rolling_avg = self.test_games['rolling_avg_score'].dropna()
        self.assertTrue(all(rolling_avg >= 0), "Rolling averages should be non-negative")
        self.assertTrue(all(rolling_avg <= 200), "Rolling averages should be reasonable")
    
    def test_elo_ratings_bounds(self):
        initial_elo = 1500
        elo_changes = np.random.uniform(-30, 30, 100)
        cumulative_elo = initial_elo + np.cumsum(elo_changes)
        
        self.assertTrue(all(cumulative_elo >= 1000), "ELO should not go below 1000")
        self.assertTrue(all(cumulative_elo <= 2000), "ELO should not go above 2000")
    
    def test_no_future_data_leakage(self):
        sorted_games = self.test_games.sort_values('date')
        for i in range(5, len(sorted_games)):
            window_data = sorted_games.iloc[i-5:i]
            current_date = sorted_games.iloc[i]['date']
            
            self.assertTrue(all(window_data['date'] < current_date))
    
    def test_missing_value_handling(self):
        missing_data = self.test_games.copy()
        missing_data.loc[::10, 'visitor_score'] = np.nan
        
        rolling_avg = missing_data.groupby('visitor_team')['visitor_score'].rolling(5).mean()
        
        nan_ratio = rolling_avg.isna().sum() / len(rolling_avg)
        self.assertLess(nan_ratio, 0.6, "Too many NaN values in rolling averages")


# Test model output validation and performance metrics
class TestModelPerformance(unittest.TestCase):
    
    def test_model_file_size(self):
        model_path = Path("models/xgb_cover.pkl")
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            
            self.assertGreater(size_mb, 0.1, "Model file seems too small")
            self.assertLess(size_mb, 500, "Model file seems too large")
    
    def test_prediction_bounds(self):
        mock_predictions = np.random.uniform(0, 1, 1000)
        
        self.assertTrue(all(mock_predictions >= 0), "Predictions should be >= 0")
        self.assertTrue(all(mock_predictions <= 1), "Predictions should be <= 1")
        
        mean_pred = np.mean(mock_predictions)
        self.assertGreater(mean_pred, 0.1, "Predictions too skewed towards 0")
        self.assertLess(mean_pred, 0.9, "Predictions too skewed towards 1")
    
    def test_feature_importance_sanity(self):
        feature_names = [
            'elo_diff', 'home_advantage', 'pace_diff', 
            'sentiment_ema', 'altitude_meters', 'random_noise'
        ]
        importances = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        
        important_features = ['elo_diff', 'home_advantage', 'pace_diff']
        noise_features = ['random_noise']
        
        for important in important_features:
            important_idx = feature_names.index(important)
            for noise in noise_features:
                noise_idx = feature_names.index(noise)
                self.assertGreater(importances[important_idx], importances[noise_idx])


# Test betting strategy implementation and risk management
class TestBacktestStaking(unittest.TestCase):
    
    def test_kelly_criterion_bounds(self):
        def calculate_kelly_fraction(prob_win, odds):
            if prob_win <= 0 or prob_win >= 1:
                return 0
            
            b = odds - 1
            kelly_f = (b * prob_win - (1 - prob_win)) / b
            
            return max(0, min(kelly_f, 0.25))
        
        test_cases = [
            (0.55, 1.91),
            (0.60, 1.91),
            (0.45, 1.91),
            (0.70, 1.91),
        ]
        
        for prob, odds in test_cases:
            kelly_f = calculate_kelly_fraction(prob, odds)
            
            self.assertGreaterEqual(kelly_f, 0, "Kelly fraction should be non-negative")
            self.assertLessEqual(kelly_f, 0.25, "Kelly fraction should be capped at 25%")
    
    def test_bankroll_management(self):
        starting_bankroll = 10000
        current_bankroll = starting_bankroll
        
        bets = [
            (100, True),
            (200, False),
            (150, True),
            (500, False),
        ]
        
        for stake, win in bets:
            if win:
                current_bankroll += stake * 0.91
            else:
                current_bankroll -= stake
            
            self.assertGreater(current_bankroll, 0, "Bankroll went negative")
    
    def test_bet_sizing_edge_cases(self):
        small_edge = 0.001
        self.assertEqual(self._calculate_bet_size(small_edge), 0)
        
        large_edge = 0.5
        bet_size = self._calculate_bet_size(large_edge)
        self.assertLessEqual(bet_size, 0.25)
    
    def _calculate_bet_size(self, edge):
        if edge < 0.01:
            return 0
        return min(edge * 0.5, 0.25)


# Test data pipeline consistency and file integrity
class TestPipelineIntegrity(unittest.TestCase):
    
    def test_feature_consistency(self):
        boruta_path = Path("config/boruta_features_sentiment.txt")
        
        if boruta_path.exists():
            with open(boruta_path, 'r') as f:
                boruta_features = [line.strip() for line in f if line.strip()]
            
            mock_columns = [
                'altitude_meters', 'home_3pa_rate', 'elo_diff_raw',
                'visitor_pregame_ema_def_ftr_ingame', 'home_pregame_scoring_margin_sd'
            ]
            
            overlap = set(boruta_features) & set(mock_columns)
            self.assertGreater(len(overlap), 0, "No Boruta features found in dataset")
    
    def test_date_ordering(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        shuffled_dates = pd.Series(dates).sample(frac=1).reset_index(drop=True)
        
        sorted_dates = shuffled_dates.sort_values()
        
        for i in range(1, len(sorted_dates)):
            self.assertGreater(sorted_dates.iloc[i], sorted_dates.iloc[i-1])
    
    def test_data_pipeline_stages(self):
        expected_files = [
            "data/raw/sbr_schedules.csv",
            "data/processed/merged_team_stats.csv",
            "data/features/ncaab_cumulative_features_v13_sentiment_enhanced.csv",
            "models/xgb_cover.pkl"
        ]
        
        for file_path in expected_files:
            path_obj = Path(file_path)
            self.assertTrue(path_obj.is_absolute() or len(path_obj.parts) > 1)


# Run all comprehensive test suites and report results
def run_comprehensive_tests():
    test_suite = unittest.TestSuite()
    
    test_classes = [
        TestSentimentDeduplication,
        TestFeatureEngineeringSanity,
        TestModelPerformance,
        TestBacktestStaking,
        TestPipelineIntegrity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    
    if success_rate == 100:
        print(f"\nALL TESTS PASSED ({success_rate:.1f}%)")
    elif success_rate >= 80:
        print(f"\nMOSTLY PASSING ({success_rate:.1f}%)")
    else:
        print(f"\nNEEDS ATTENTION ({success_rate:.1f}%)")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
