# Student Performance Prediction

## Astana IT University - Educational Practice Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## 📋 Project Overview

This project implements machine learning models to predict university student performance based on their study habits, lifestyle factors, and demographic information. The system analyzes a comprehensive dataset of 10,000 global university students to identify key factors influencing academic success.

### 🎯 Objectives

- Analyze relationships between student habits and academic performance
- Build predictive models for final exam scores and assignment performance
- Identify key factors contributing to student success
- Provide insights for educational improvement strategies

## 📊 Dataset

**File:** `global_university_students_performance_habits_10000.csv`

**Size:** 10,000 students from multiple countries

### Features (26 columns):

#### Demographic Information
- `student_id`: Unique identifier
- `age`: Student age (18-28)
- `gender`: Male/Female
- `country`: Country of study
- `major`: Field of study
- `university_year`: Year level (1-4)

#### Academic Metrics
- `GPA`: Grade Point Average (0.0-4.0)
- `class_attendance_percent`: Attendance percentage
- `final_exam_score`: **Target variable** (0-100)
- `assignment_score`: Assignment performance (0-100)

#### Study Habits
- `study_hours_per_day`: Daily study time
- `exam_preparation_days`: Days preparing for exams
- `note_taking_method`: Handwritten/Digital/Mixed
- `AI_tool_usage_hours`: Hours using AI tools
- `favorite_AI_tool`: ChatGPT/Claude/Copilot/Gemini/None

#### Lifestyle Factors
- `sleep_hours`: Average sleep per night
- `screen_time_hours`: Daily screen time
- `social_media_hours`: Daily social media usage
- `gaming_hours`: Daily gaming time
- `exercise_hours_per_week`: Weekly exercise
- `coffee_consumption_per_day`: Daily coffee intake
- `extracurricular_hours_per_week`: Time in extracurricular activities

#### Other Factors
- `part_time_job`: Yes/No
- `relationship_status`: Single/In a Relationship
- `family_income_level`: Low/Middle/High
- `internet_quality`: Poor/Average/Good
- `mental_stress_level`: Stress level (1-10)

## 🏗️ Project Structure

```
Project/
├── data/
│   └── global_university_students_performance_habits_10000.csv
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── train_model.py          # Model training script
│   └── predict.py               # Inference script
├── models/                      # Saved trained models
├── results/                     # Visualizations and results
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

This notebook provides:
- Dataset overview and statistics
- Feature distributions and correlations
- Study habits vs performance analysis
- Lifestyle factors impact
- AI tool usage insights
- Mental stress analysis

### 2. Train Models

Run the training script to build and evaluate multiple ML models:
```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Train 8 different regression models
- Evaluate model performance
- Generate visualization plots in `results/`
- Save the best model to `models/`

### 3. Make Predictions

Use the trained model for predictions:
```bash
python src/predict.py
```

Or import in your own code:
```python
from src.predict import load_model_and_preprocessors, predict_single_student

# Load model
model, scaler, encoders = load_model_and_preprocessors()

# Predict for a student
student_data = {
    'age': 22,
    'gender': 'Male',
    'country': 'USA',
    'major': 'Computer Science',
    'GPA': 3.5,
    'study_hours_per_day': 4.5,
    # ... other features
}

predicted_score = predict_single_student(student_data, model, scaler, encoders)
print(f"Predicted Final Exam Score: {predicted_score:.2f}")
```

## 🤖 Models

The project implements and compares the following regression models:

1. **Linear Regression** - Baseline model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization
4. **Decision Tree** - Non-linear relationships
5. **Random Forest** - Ensemble of decision trees
6. **Gradient Boosting** - Sequential ensemble
7. **XGBoost** - Optimized gradient boosting
8. **LightGBM** - Fast gradient boosting

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)
- **MSE** (Mean Squared Error)

## 📊 Expected Results

The models typically achieve:
- **R² Score**: 0.85-0.95 (depending on model)
- **RMSE**: 3-8 points
- **MAE**: 2-6 points

Key findings:
- GPA is the strongest predictor of final exam performance
- Study hours and class attendance show positive correlation
- Mental stress level has negative impact on performance
- AI tool usage shows varying effects by tool type

## 🔍 Key Insights

From the exploratory analysis:

1. **Study Habits Matter**: Students who study 4+ hours daily and prepare 2+ weeks for exams score significantly higher
2. **Attendance Impact**: 90%+ attendance correlates with better performance
3. **Sleep is Critical**: 7-8 hours of sleep shows optimal results
4. **AI Tool Usage**: Moderate AI tool usage (1-2 hours) shows positive correlation
5. **Stress Effects**: High mental stress (7+) negatively impacts scores
6. **Major Differences**: STEM majors show different patterns than humanities

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML models and preprocessing
- **XGBoost** - Gradient boosting
- **LightGBM** - Gradient boosting
- **matplotlib/seaborn** - Visualization
- **Jupyter** - Interactive analysis

## 📝 Future Improvements

- [ ] Implement deep learning models (Neural Networks)
- [ ] Add feature engineering for interaction terms
- [ ] Build a web interface for predictions
- [ ] Integrate real-time student tracking
- [ ] Add explainability with SHAP values
- [ ] Implement multi-target prediction (both exam and assignment scores)

## 👥 Contributors

- **Zhaksylyk Seizhan** - Astana IT University
- **Altemir Sailaukhan** - Astana IT University
- **Chingiz Nadirov** - Astana IT University
- **Supervisor:** Ali Sapabek ([GitHub](https://github.com/SapabekAli))

## 📄 License

This project is part of the Educational Practice at Astana IT University.

## 🙏 Acknowledgments

- Astana IT University for providing the opportunity
- Dataset source: Global University Students Performance and Habits
- Open-source ML libraries community

## 📧 Contact

For questions or feedback:
- Students: Zhaksylyk Seizhan, Altemir Sailaukhan, Chingiz Nadirov
- Supervisor: Ali Sapabek
- Institution: Astana IT University

---

**Astana IT University** | Educational Practice Project | 2024
