"""
Student Performance Prediction - Inference Script
Astana IT University - Educational Practice

This script loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os


def load_model_and_preprocessors(model_dir='models'):
    """Load the trained model and preprocessing objects"""
    print("Loading model and preprocessors...")

    # Find the best model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No trained model found. Please run train_model.py first.")

    model_path = os.path.join(model_dir, model_files[0])
    model = joblib.load(model_path)

    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))

    print(f"[OK] Model loaded: {model_files[0]}")
    return model, scaler, label_encoders


def preprocess_input(data, scaler, label_encoders):
    """Preprocess input data for prediction"""
    df = data.copy()

    # Drop columns not used in training
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
    if 'final_exam_score' in df.columns:
        df = df.drop('final_exam_score', axis=1)
    if 'assignment_score' in df.columns:
        df = df.drop('assignment_score', axis=1)

    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    # Scale features
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index
    )

    return df_scaled


def predict_performance(input_data, model, scaler, label_encoders):
    """Make predictions on input data"""
    X = preprocess_input(input_data, scaler, label_encoders)
    predictions = model.predict(X)
    return predictions


def predict_single_student(student_features, model, scaler, label_encoders):
    """Predict performance for a single student"""
    df = pd.DataFrame([student_features])
    prediction = predict_performance(df, model, scaler, label_encoders)
    return prediction[0]


def main():
    """Example usage"""
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION - INFERENCE")
    print("="*60)

    # Load model and preprocessors
    model, scaler, label_encoders = load_model_and_preprocessors()

    # Example: Load test data
    print("\nLoading test data...")
    df = pd.read_csv('data/global_university_students_performance_habits_10000.csv')

    # Use first 5 students as example
    test_students = df.head(5).copy()
    actual_scores = test_students['final_exam_score'].values

    # Make predictions
    predictions = predict_performance(test_students, model, scaler, label_encoders)

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS (First 5 students)")
    print("="*60)

    results = pd.DataFrame({
        'Student ID': test_students['student_id'].values,
        'Predicted Score': predictions.round(2),
        'Actual Score': actual_scores,
        'Difference': (predictions - actual_scores).round(2)
    })

    print("\n", results.to_string(index=False))

    # Example: Single student prediction
    print("\n" + "="*60)
    print("EXAMPLE: Single Student Prediction")
    print("="*60)

    example_student = {
        'age': 22,
        'gender': 'Male',
        'country': 'USA',
        'major': 'Computer Science',
        'university_year': 3,
        'GPA': 3.5,
        'study_hours_per_day': 4.5,
        'class_attendance_percent': 95,
        'sleep_hours': 7.0,
        'screen_time_hours': 5.0,
        'social_media_hours': 2.0,
        'gaming_hours': 1.5,
        'exercise_hours_per_week': 3.0,
        'part_time_job': 'No',
        'relationship_status': 'Single',
        'family_income_level': 'Middle',
        'internet_quality': 'Good',
        'mental_stress_level': 4.0,
        'AI_tool_usage_hours': 2.0,
        'favorite_AI_tool': 'ChatGPT',
        'note_taking_method': 'Digital',
        'exam_preparation_days': 14,
        'coffee_consumption_per_day': 2,
        'extracurricular_hours_per_week': 5.0
    }

    prediction = predict_single_student(example_student, model, scaler, label_encoders)

    print("\nStudent Profile:")
    for key, value in example_student.items():
        print(f"  {key}: {value}")

    print(f"\n[PREDICTION] Predicted Final Exam Score: {prediction:.2f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
