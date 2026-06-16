# Results Directory

This directory will contain visualizations and analysis results after running the training scripts.

## Files Generated

After training, you'll find:
- `model_comparison.png` - Performance comparison of all models
- `actual_vs_predicted.png` - Scatter plot of predictions vs actual scores
- `feature_importance.png` - Top features contributing to predictions

## Usage

Visualizations are automatically generated when running:
```bash
python src/train_model.py
```

Or from notebooks:
```python
from train_model import StudentPerformancePredictor

predictor = StudentPerformancePredictor('data/dataset.csv')
predictor.load_data()
predictor.preprocess_data()
predictor.train_models()
predictor.plot_results(save_path='results/')
```

## Note

PNG files are excluded from git (see .gitignore).
Generate your own visualizations by running the training pipeline.
