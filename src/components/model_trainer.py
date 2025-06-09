import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.util import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0],
                    'max_iter': [1000]
                },
                "XGBClassifier": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5]
                },
                "CatBoosting Classifier": {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 200]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train,
                                        X_test=X_test, y_test=y_test,
                                        models=models, param=params)
            
            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Train the best model with the best parameters
            best_params = params[best_model_name]
            gs = GridSearchCV(best_model, best_params, cv=3)
            gs.fit(X_train, y_train)
            
            final_model = gs.best_estimator_
            final_model.fit(X_train, y_train)

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )

            print(f"\nBest model ({best_model_name}) has been saved to: {os.path.abspath(self.model_trainer_config.trained_model_file_path)}")
            print(f"Model Accuracy: {best_model_score:.2%}\n")

            predicted = final_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
