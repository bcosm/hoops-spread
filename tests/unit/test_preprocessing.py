
# Unit tests for data preprocessing functions including cleaning, standardization, and validation
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

import pytest, importlib
if not importlib.util.find_spec("hoops_spread.preprocessing"):
    pytest.skip("preprocessing module not implemented yet", allow_module_level=True)


# Test class for individual preprocessing functions with sample data scenarios
class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'team': ['Duke', 'UNC', 'Kentucky', 'Louisville'],
            'points': [75.5, 82.1, np.nan, 70.2],
            'rebounds': [35, np.nan, 40, 38],
            'assists': [18, 15, 12, np.nan]
        })
        
        self.sample_schedule = pd.DataFrame({
            'date': ['2022-11-15', '2022-11-16', '2022-11-17'],
            'away_team': ['Duke', 'UNC', 'Kentucky'],
            'home_team': ['Virginia', 'NC State', 'Louisville'],
            'away_score': [75, 80, 85],
            'home_score': [78, 82, 83]
        })
    
    def test_clean_missing_values_drop_rows(self):
        result = clean_missing_values(
            self.sample_data, 
            strategy='drop_rows',
            threshold=0.5
        )
        
        self.assertLess(len(result), len(self.sample_data))
        self.assertFalse(result.isnull().any().any())
    
    def test_clean_missing_values_impute_mean(self):
        result = clean_missing_values(
            self.sample_data, 
            strategy='impute_mean',
            columns=['points', 'rebounds', 'assists']
        )
        
        self.assertFalse(result[['points', 'rebounds', 'assists']].isnull().any().any())
        
        original_points_mean = self.sample_data['points'].mean()
        self.assertAlmostEqual(result.loc[2, 'points'], original_points_mean, places=2)
    
    def test_clean_missing_values_impute_median(self):
        result = clean_missing_values(
            self.sample_data, 
            strategy='impute_median',
            columns=['rebounds']
        )
        
        original_rebounds_median = self.sample_data['rebounds'].median()
        self.assertEqual(result.loc[1, 'rebounds'], original_rebounds_median)
    
    def test_standardize_team_names(self):
        messy_names = pd.DataFrame({
            'team': ['  Duke  ', 'north carolina', 'KENTUCKY', 'N.C. State'],
            'points': [75, 80, 85, 70]
        })
        
        result = standardize_team_names(messy_names)
        
        expected_names = ['Duke', 'North Carolina', 'Kentucky', 'NC State']
        self.assertEqual(result['team'].tolist(), expected_names)
    
    def test_merge_datasets_inner(self):
        left_df = pd.DataFrame({
            'team': ['Duke', 'UNC', 'Kentucky'],
            'conference': ['ACC', 'ACC', 'SEC']
        })
        
        right_df = pd.DataFrame({
            'team': ['Duke', 'UNC', 'Virginia'],
            'ranking': [5, 8, 12]
        })
        
        result = merge_datasets(left_df, right_df, on='team', how='inner')        
        self.assertEqual(len(result), 2)
        self.assertIn('conference', result.columns)
        self.assertIn('ranking', result.columns)
    
    def test_merge_datasets_left(self):
        left_df = pd.DataFrame({
            'team': ['Duke', 'UNC', 'Kentucky'],
            'conference': ['ACC', 'ACC', 'SEC']
        })
        
        right_df = pd.DataFrame({
            'team': ['Duke', 'UNC'],
            'ranking': [5, 8]
        })
        
        result = merge_datasets(left_df, right_df, on='team', how='left')
        
        self.assertEqual(len(result), 3)
        self.assertTrue(pd.isna(result.loc[result['team'] == 'Kentucky', 'ranking'].iloc[0]))
    
    def test_validate_data_quality(self):
        good_data = pd.DataFrame({
            'team': ['Duke', 'UNC', 'Kentucky'],
            'points': [75.5, 82.1, 78.3],
            'date': pd.to_datetime(['2022-11-15', '2022-11-16', '2022-11-17'])
        })
        
        issues = validate_data_quality(good_data)
        self.assertEqual(len(issues), 0)
        
        bad_data = pd.DataFrame({
            'team': ['Duke', 'UNC', '', 'Kentucky'],
            'points': [75.5, np.nan, 82.1, -10],
            'date': ['2022-11-15', 'invalid-date', '2022-11-17', '2022-11-18']
        })
        
        issues = validate_data_quality(bad_data)
        self.assertGreater(len(issues), 0)
    
    def test_preprocess_schedules(self):
        raw_schedule = pd.DataFrame({
            'Date': ['Nov 15, 2022', 'Nov 16, 2022'],
            'Away': ['  Duke  ', 'north carolina'],
            'Home': ['Virginia', 'NC STATE'],
            'Away_Score': ['75', '80'],
            'Home_Score': ['78', '82']
        })
        
        result = preprocess_schedules(raw_schedule)
        
        expected_columns = ['date', 'away_team', 'home_team', 'away_score', 'home_score']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        self.assertEqual(result['date'].dtype, 'datetime64[ns]')
        
        self.assertEqual(result['away_team'].iloc[0], 'Duke')
        self.assertEqual(result['away_team'].iloc[1], 'North Carolina')
        self.assertEqual(result['home_team'].iloc[1], 'NC State')
        
        self.assertEqual(result['away_score'].dtype, 'int64')
        self.assertEqual(result['home_score'].dtype, 'int64')


# Integration test class for complete preprocessing pipeline with end-to-end data transformation
class TestPreprocessingIntegration(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_preprocessing_pipeline(self):
        raw_data = pd.DataFrame({
            'Date': ['Nov 15, 2022', 'Nov 16, 2022', 'Nov 17, 2022'],
            'Away': ['  Duke  ', 'north carolina', 'KENTUCKY'],
            'Home': ['Virginia', 'NC STATE', 'louisville'],
            'Away_Score': ['75', '80', ''],
            'Home_Score': ['78', '82', '83']
        })
        
        processed_schedules = preprocess_schedules(raw_data)
        
        cleaned_data = clean_missing_values(
            processed_schedules, 
            strategy='drop_rows',
            threshold=0.8
        )
        
        issues = validate_data_quality(cleaned_data)
        
        self.assertGreater(len(processed_schedules), 0)
        self.assertEqual(len(cleaned_data), 2)
        self.assertEqual(len(issues), 0)


if __name__ == '__main__':
    unittest.main()
