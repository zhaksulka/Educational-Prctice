"""
Student Performance Prediction - Machine Learning Models
Astana IT University - Educational Practice

This script trains and evaluates multiple ML models for predicting student performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib
import os

warnings.filterwarnings('ignore')


class StudentPerformancePredictor:
    """Machine Learning pipeline for student performance prediction"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nMissing values: {self.df.isnull().sum().sum()}")
        return self.df

    def preprocess_data(self, target='final_exam_score'):
        """Preprocess data: encode categorical variables and scale features"""
        print(f"\nPreprocessing data for target: {target}")

        df_processed = self.df.copy()

        # Drop student_id as it's not a feature
        if 'student_id' in df_processed.columns:
            df_processed = df_processed.drop('student_id', axis=1)

        # Separate features and target
        X = df_processed.drop(['final_exam_score', 'assignment_score'], axis=1)
        y = df_processed[target]

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Encoding {len(categorical_cols)} categorical features...")

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """Train multiple regression models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        }

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

            # Predictions
            y_pred = model.predict(self.X_test)

            # Metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            self.results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R^2 Score: {r2:.4f}")

    def evaluate_models(self):
        """Display comprehensive evaluation of all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)

        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('R2', ascending=False)

        print("\n", results_df.to_string())

        # Find best model
        best_model_name = results_df['R2'].idxmax()
        print(f"\n[BEST] Best Model: {best_model_name}")
        print(f"   R^2 Score: {results_df.loc[best_model_name, 'R2']:.4f}")
        print(f"   RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}")

        return results_df, best_model_name

    def plot_results(self, save_path='results'):
        """Create visualization of model performance"""
        os.makedirs(save_path, exist_ok=True)

        results_df = pd.DataFrame(self.results).T

        # Plot 1: Model Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        for idx, metric in enumerate(metrics):
            row, col = idx // 2, idx % 2
            results_df[metric].sort_values().plot(kind='barh', ax=axes[row, col], color='skyblue')
            axes[row, col].set_title(f'Model Comparison - {metric}', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel(metric)
            axes[row, col].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[SAVED] {save_path}/model_comparison.png")
        plt.close()

        # Plot 2: Actual vs Predicted for best model
        best_model_name = results_df['R2'].idxmax()
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(self.X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Score', fontsize=12)
        plt.ylabel('Predicted Score', fontsize=12)
        plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}/actual_vs_predicted.png")
        plt.close()

        # Plot 3: Feature Importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'], color='coral')
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top 15 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"[SAVED] {save_path}/feature_importance.png")
            plt.close()

    def save_best_model(self, save_path='models'):
        """Save the best performing model"""
        os.makedirs(save_path, exist_ok=True)

        results_df = pd.DataFrame(self.results).T
        best_model_name = results_df['R2'].idxmax()
        best_model = self.models[best_model_name]

        model_path = f'{save_path}/best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
        joblib.dump(best_model, model_path)
        print(f"\n[SAVED] Best model saved: {model_path}")

        # Save scaler and encoders
        joblib.dump(self.scaler, f'{save_path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{save_path}/label_encoders.pkl')
        print(f"[SAVED] Preprocessing objects saved")

        return model_path


def main():
    """Main execution function"""
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION")
    print("Astana IT University - Educational Practice")
    print("="*60)

    # Initialize predictor
    predictor = StudentPerformancePredictor('data/global_university_students_performance_habits_10000.csv')

    # Load data
    predictor.load_data()

    # Preprocess data (predicting final_exam_score)
    predictor.preprocess_data(target='final_exam_score')

    # Train models
    predictor.train_models()

    # Evaluate models
    results_df, best_model = predictor.evaluate_models()

    # Plot results
    predictor.plot_results()

    # Save best model
    predictor.save_best_model()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check results/ folder for visualizations")
    print("2. Review models/ folder for saved models")
    print("3. Run predictions using the saved model")


if __name__ == "__main__":
    main()
