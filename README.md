# ChurnMaster Pro: ML-Powered Customer Retention System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-Latest-orange)
![Flask](https://img.shields.io/badge/Flask-Latest-lightgrey)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-brightgreen)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🎯 Overview

ChurnMaster Pro is a production-ready machine learning system that predicts customer churn in the telecommunications industry. Built with Python and modern ML frameworks, it features a user-friendly web interface for real-time predictions.

### 🚀 Key Features

- **Advanced ML Pipeline**: Automated data preprocessing, model training, and hyperparameter tuning
- **Multiple ML Models**: Implements Random Forest, XGBoost, CatBoost, and more
- **Interactive Web UI**: Flask-based interface for instant predictions
- **Production-Ready**: Includes logging, error handling, and modular code structure

## 🛠️ Tech Stack

- **Machine Learning**: Scikit-learn, XGBoost, CatBoost
- **Web Framework**: Flask
- **Frontend**: Bootstrap 5
- **Data Processing**: Pandas, NumPy
- **Development**: Python 3.7+

## 📂 Project Structure

```
├── artifacts/           # Trained models and preprocessors
├── logs/               # Application logs
├── notebook/           # Jupyter notebooks (EDA & Development)
│   └── data/          # Dataset files
├── src/               # Source code
│   ├── components/    # Core ML components
│   ├── pipeline/      # Training & prediction pipelines
│   └── utils/         # Helper functions
├── templates/         # Flask HTML templates
├── app.py            # Flask application
└── requirements.txt   # Dependencies
```

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd churnmaster-pro
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **Train the Model**
   ```bash
   python src/pipeline/train_pipeline.py
   ```

4. **Run the Web App**
   ```bash
   python app.py
   ```

5. **Access the Interface**
   ```
   http://localhost:5000
   ```

## 🔧 Model Features

### Input Features
- **Customer Demographics**: Gender, age, partner status
- **Service Details**: Internet type, phone service, security
- **Contract Info**: Type, tenure, charges
- **Payment Data**: Method, monthly charges, total charges

### Target
- Customer Churn (Binary Classification)

## 📊 Model Performance

The system automatically:
- Evaluates multiple ML algorithms
- Performs cross-validation
- Tunes hyperparameters
- Selects the best performing model

## 🌟 Features in Detail

### Data Processing
- Automated data cleaning
- Advanced feature engineering
- Handles missing values
- Categorical encoding
- Feature scaling

### Model Training
- Cross-validation
- Hyperparameter optimization
- Model selection
- Performance metrics

### Web Interface
- User-friendly form
- Real-time predictions
- Responsive design
- Clear result display

## 📝 Logging & Monitoring

- Comprehensive logging system
- Performance tracking
- Error monitoring
- Training history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- Telco Customer Churn dataset
- Scikit-learn community
- Flask framework

---
⭐️ If you find this project useful, please consider giving it a star!