"""
Data Preprocessing Utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(df):
    """Clean and validate dataset"""
    df_clean = df.copy()

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Handle missing values if any
    if df_clean.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # Fill categorical columns with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    return df_clean


def create_features(df):
    """Create additional features for better predictions"""
    df_featured = df.copy()

    # Total screen time
    df_featured['total_screen_time'] = (
        df_featured['screen_time_hours'] +
        df_featured['social_media_hours'] +
        df_featured['gaming_hours']
    )

    # Study to leisure ratio
    df_featured['study_leisure_ratio'] = (
        df_featured['study_hours_per_day'] /
        (df_featured['total_screen_time'] + 1)  # +1 to avoid division by zero
    )

    # Overall wellness score
    df_featured['wellness_score'] = (
        df_featured['sleep_hours'] +
        df_featured['exercise_hours_per_week'] / 7 -
        df_featured['mental_stress_level']
    )

    # Academic engagement
    df_featured['academic_engagement'] = (
        df_featured['study_hours_per_day'] *
        df_featured['class_attendance_percent'] / 100
    )

    # Binary features
    df_featured['high_gpa'] = (df_featured['GPA'] >= 3.5).astype(int)
    df_featured['uses_ai'] = (df_featured['AI_tool_usage_hours'] > 0).astype(int)

    return df_featured


def get_feature_groups():
    """Return feature groups for analysis"""
    return {
        'demographic': ['age', 'gender', 'country', 'major', 'university_year'],
        'academic': ['GPA', 'class_attendance_percent', 'study_hours_per_day',
                    'exam_preparation_days', 'note_taking_method'],
        'lifestyle': ['sleep_hours', 'screen_time_hours', 'social_media_hours',
                     'gaming_hours', 'exercise_hours_per_week', 'coffee_consumption_per_day'],
        'ai_usage': ['AI_tool_usage_hours', 'favorite_AI_tool'],
        'social': ['part_time_job', 'relationship_status', 'family_income_level',
                  'extracurricular_hours_per_week'],
        'environmental': ['internet_quality', 'mental_stress_level']
    }
