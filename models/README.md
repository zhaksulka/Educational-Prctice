# Models Directory

This directory will contain trained machine learning models after running `train_model.py`.

## Files Generated

After training, you'll find:
- `best_model_*.pkl` - The best performing model
- `scaler.pkl` - Feature scaler object
- `label_encoders.pkl` - Categorical encoders

## Usage

Load models for predictions:
```python
import joblib

model = joblib.load('models/best_model_lightgbm.pkl')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')
```

## Note

Model files are large and excluded from git (see .gitignore).
Train your own models by running: `python src/train_model.py`
