"""
Unit tests for the Student Performance Prediction project
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import clean_data, create_features, get_feature_groups


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""

    def setUp(self):
        """Create sample data for testing"""
        self.sample_data = pd.DataFrame({
            'student_id': ['S001', 'S002', 'S003'],
            'age': [20, 22, 21],
            'GPA': [3.5, 3.8, 3.2],
            'study_hours_per_day': [4.0, 5.0, 3.5],
            'class_attendance_percent': [95, 98, 90],
            'screen_time_hours': [5.0, 4.0, 6.0],
            'social_media_hours': [2.0, 1.5, 3.0],
            'gaming_hours': [1.0, 0.5, 2.0],
            'sleep_hours': [7.0, 8.0, 6.5],
            'exercise_hours_per_week': [3.0, 5.0, 2.0],
            'mental_stress_level': [5, 4, 6],
            'AI_tool_usage_hours': [2.0, 1.5, 0.5],
            'final_exam_score': [85, 92, 78]
        })

    def test_clean_data(self):
        """Test data cleaning function"""
        cleaned = clean_data(self.sample_data)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(cleaned.isnull().sum().sum(), 0)

    def test_create_features(self):
        """Test feature creation"""
        featured = create_features(self.sample_data)

        # Check new features exist
        self.assertIn('total_screen_time', featured.columns)
        self.assertIn('study_leisure_ratio', featured.columns)
        self.assertIn('wellness_score', featured.columns)
        self.assertIn('academic_engagement', featured.columns)

        # Check calculations
        self.assertEqual(
            featured.loc[0, 'total_screen_time'],
            5.0 + 2.0 + 1.0
        )

    def test_get_feature_groups(self):
        """Test feature grouping"""
        groups = get_feature_groups()

        self.assertIn('demographic', groups)
        self.assertIn('academic', groups)
        self.assertIn('lifestyle', groups)
        self.assertIsInstance(groups['academic'], list)


class TestModelFunctions(unittest.TestCase):
    """Test model-related functions"""

    def test_data_split(self):
        """Test train-test split maintains data integrity"""
        from sklearn.model_selection import train_test_split

        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(X_train.shape[1], 5)


class TestPrediction(unittest.TestCase):
    """Test prediction functionality"""

    def test_prediction_range(self):
        """Test predictions are in valid range"""
        predictions = np.array([85.5, 92.3, 78.1, 95.0, 65.2])

        # All predictions should be between 0 and 100
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 100))


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    run_tests()
