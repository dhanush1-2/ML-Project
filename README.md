# Customer Churn Prediction Project

This project is an end-to-end machine learning solution for predicting customer churn in a telecommunications company. It includes data preprocessing, model training, and a web interface for making predictions.

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
├── templates/         # Flask HTML templates
│   ├── base.html     # Base template with common styling
│   ├── index.html    # Landing page
│   └── home.html     # Prediction form
├── app.py            # Flask application
├── requirements.txt   # Project dependencies
└── setup.py          # Package configuration
```

## Features

- **Data Processing**:
  - Automated data ingestion with train-test split
  - Advanced preprocessing pipeline:
    - Numerical features: Standard scaling with mean imputation
    - Categorical features: One-hot encoding and label encoding
    - Automated handling of missing values
  
- **Model Training**:
  - Multiple algorithm evaluation:
    - Random Forest
    - Decision Tree
    - Gradient Boosting
    - Logistic Regression
    - XGBoost
    - CatBoost
    - AdaBoost
  - Automated hyperparameter tuning using GridSearchCV
  - Best model selection based on accuracy
  - Model persistence for later use

- **Web Interface**:
  - User-friendly Flask web application
  - Interactive form for entering customer data
  - Real-time predictions
  - Clean, responsive design using Bootstrap
  - Clear visualization of prediction results

- **Code Quality**:
  - Modular and maintainable code structure
  - Custom exception handling
  - Comprehensive logging
  - Type hints and documentation

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:
   ```bash
   python src/pipeline/train_pipeline.py
   ```
   This will:
   - Load and preprocess the data
   - Train multiple models
   - Select the best performing model
   - Save the model and preprocessor in the artifacts directory
   
   You'll see detailed progress output and final model accuracy.

2. Start the web application:
   ```bash
   python app.py
   ```
   
3. Open your browser and go to:
   ```
   http://localhost:5000
   ```

4. Use the web interface to:
   - Enter customer information through the form
   - Get instant churn predictions
   - View prediction results with clear visual feedback

## Data

The project uses the Telco Customer Churn dataset with the following features:

### Customer Information:
- Demographics (gender, senior citizen status)
- Account details (tenure, contract type)
- Services subscribed (phone, internet, security, etc.)
- Billing information (payment method, monthly charges)

### Target Variable:
- Churn (Yes/No): Whether the customer left the company

## Model Performance

The training pipeline:
1. Evaluates multiple models with cross-validation
2. Performs hyperparameter tuning for each model
3. Selects the best performing model based on accuracy
4. Provides detailed logging of model performance metrics

The final selected model achieves competitive performance in predicting customer churn, with accuracy metrics displayed during training.

## Logging and Monitoring

- All training steps are logged in the `logs` directory
- Each run creates a timestamped log file
- Logs include:
  - Data preprocessing steps
  - Model training progress
  - Performance metrics
  - Error messages (if any)

## Error Handling

The project includes robust error handling:
- Custom exception classes for different types of errors
- Detailed error messages with line numbers
- Error logging for debugging
- User-friendly error messages in the web interface

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.