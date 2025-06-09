# Customer Churn Prediction Project

This project is an end-to-end machine learning solution for predicting customer churn in a telecommunications company. It includes data preprocessing, model training, and a prediction pipeline.

## Project Structure

```
├── artifacts/           # Stores trained models and preprocessors
├── logs/               # Logging files
├── notebook/           # Jupyter notebooks for EDA and model development
│   └── data/          # Raw data files
├── src/               # Source code
│   ├── components/    # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/      # Training and prediction pipelines
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py   # Custom exception handling
│   ├── logger.py      # Logging configuration
│   └── util.py        # Utility functions
├── requirements.txt    # Project dependencies
└── setup.py           # Package configuration
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the model:
   ```
   python src/pipeline/train_pipeline.py
   ```

2. The trained model and preprocessor will be saved in the `artifacts` directory.

3. The model can then be used for predictions using the prediction pipeline.

## Features

- Data ingestion with train-test split
- Advanced data preprocessing with numerical and categorical pipelines
- Model selection from multiple algorithms:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Logistic Regression
  - XGBoost
  - CatBoost
  - AdaBoost
- Hyperparameter tuning using GridSearchCV
- Custom exception handling and logging
- Modular and maintainable code structure

## Data

The project uses the Telco Customer Churn dataset, which includes features like:
- Customer demographics
- Account information
- Services subscribed
- Charges
- Churn status

## Model Performance

The best performing model is selected based on accuracy score, with cross-validation during training. The final model achieves competitive performance in predicting customer churn.