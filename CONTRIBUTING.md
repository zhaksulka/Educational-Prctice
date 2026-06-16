# Contributing to Student Performance Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and environment details
- Error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- Clear description of the enhancement
- Use case and benefits
- Possible implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure CI/CD checks pass

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Formatting

We use the following tools:
- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run before committing:
```bash
black src/
isort src/
flake8 src/
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README if adding new features
- Comment complex logic

Example:
```python
def predict_performance(data: pd.DataFrame, model) -> np.ndarray:
    """
    Predict student performance using trained model.
    
    Args:
        data: DataFrame containing student features
        model: Trained machine learning model
        
    Returns:
        Array of predicted scores
        
    Raises:
        ValueError: If required features are missing
    """
    pass
```

### Testing

- Write unit tests for new functions
- Maintain test coverage above 80%
- Test edge cases and error conditions
- Use pytest framework

## Project Structure

```
Project/
├── data/              # Dataset files
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── config.py     # Configuration
│   ├── train_model.py    # Training pipeline
│   ├── predict.py    # Inference
│   └── utils.py      # Utilities
├── tests/            # Unit tests
├── models/           # Saved models
├── results/          # Outputs
└── .github/          # CI/CD workflows
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 isort
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

## Commit Message Guidelines

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add SHAP values for model explainability
fix: correct feature encoding for categorical variables
docs: update README with new usage examples
```

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Publishing others' private information
- Unprofessional conduct

## Questions?

Feel free to:
- Open an issue for discussion
- Contact the maintainers
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Student Performance Prediction! 🎓
