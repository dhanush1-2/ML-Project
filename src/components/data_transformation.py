import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
import pickle


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation based on different types of data
        """
        try:
            # Define which columns should be scaled vs one-hot encoded vs label encoded
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns_ohe = ['PaymentMethod', 'Contract', 'InternetService']  # columns for one-hot encoding
            categorical_columns_le = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                                    'PhoneService', 'MultipleLines', 'OnlineSecurity',
                                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                    'StreamingTV', 'StreamingMovies', 'PaperlessBilling']  # columns for label encoding

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline - One Hot Encoding
            cat_pipeline_ohe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False)),
                ]
            )

            # Categorical Pipeline - Label Encoding
            cat_pipeline_le = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", LabelEncoder()),  # Add LabelEncoder to the pipeline
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns (OHE): {categorical_columns_ohe}")
            logging.info(f"Categorical columns (LE): {categorical_columns_le}")

            # Create label encoders for each categorical column
            label_encoders = {}
            for col in categorical_columns_le:
                label_encoders[col] = LabelEncoder()

            # Combine all the pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline_ohe", cat_pipeline_ohe, categorical_columns_ohe),
                ]
            )

            return preprocessor, label_encoders, categorical_columns_le

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj, label_encoders, cat_columns_le = self.get_data_transformer_object()

            target_column_name = "Churn"
            
            # Convert target column to numeric
            train_df[target_column_name] = train_df[target_column_name].map({'Yes': 1, 'No': 0})
            test_df[target_column_name] = test_df[target_column_name].map({'Yes': 1, 'No': 0})

            # Convert SeniorCitizen to string type for consistency
            train_df['SeniorCitizen'] = train_df['SeniorCitizen'].astype(str)
            test_df['SeniorCitizen'] = test_df['SeniorCitizen'].astype(str)

            # Convert TotalCharges to numeric, handling any non-numeric values
            train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
            test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')

            # Fill missing values in TotalCharges with mean
            train_df['TotalCharges'].fillna(train_df['TotalCharges'].mean(), inplace=True)
            test_df['TotalCharges'].fillna(test_df['TotalCharges'].mean(), inplace=True)

            # First apply label encoding to categorical columns
            for column in cat_columns_le:
                label_encoders[column].fit(train_df[column])
                train_df[column] = label_encoders[column].transform(train_df[column])
                test_df[column] = label_encoders[column].transform(test_df[column])

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Fit and transform on training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            # Save both preprocessor and label encoders
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={
                    'preprocessor': preprocessing_obj,
                    'label_encoders': label_encoders,
                    'categorical_columns_le': cat_columns_le
                }
            )

            print(f"\nPreprocessor has been saved to: {os.path.abspath(self.data_transformation_config.preprocessor_obj_file_path)}\n")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
