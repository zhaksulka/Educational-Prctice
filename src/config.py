"""
Configuration file for the Student Performance Prediction project
"""

# Model parameters
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
}

# File paths
DATA_PATH = 'data/global_university_students_performance_habits_10000.csv'
MODEL_PATH = 'models/'
RESULTS_PATH = 'results/'

# Feature configuration
TARGET_VARIABLE = 'final_exam_score'
ALTERNATIVE_TARGET = 'assignment_score'

FEATURES_TO_DROP = ['student_id', 'final_exam_score', 'assignment_score']

CATEGORICAL_FEATURES = [
    'gender', 'country', 'major', 'part_time_job',
    'relationship_status', 'family_income_level',
    'internet_quality', 'favorite_AI_tool', 'note_taking_method'
]

NUMERIC_FEATURES = [
    'age', 'university_year', 'GPA', 'study_hours_per_day',
    'class_attendance_percent', 'sleep_hours', 'screen_time_hours',
    'social_media_hours', 'gaming_hours', 'exercise_hours_per_week',
    'mental_stress_level', 'AI_tool_usage_hours', 'exam_preparation_days',
    'coffee_consumption_per_day', 'extracurricular_hours_per_week'
]

# Model hyperparameters
HYPERPARAMETERS = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1],
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1],
    }
}

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 300,
    'style': 'whitegrid',
    'color_palette': 'Set2',
}
