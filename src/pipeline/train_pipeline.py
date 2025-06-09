import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    try:
        print("\n========== Starting Model Training Pipeline ==========\n")
        
        print("1. Data Ingestion Stage")
        print("------------------------")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(f"Training data saved to: {os.path.abspath(train_data_path)}")
        print(f"Testing data saved to: {os.path.abspath(test_data_path)}\n")
        
        print("2. Data Transformation Stage")
        print("----------------------------")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        print("\n3. Model Training Stage")
        print("------------------------")
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print("\n========== Model Training Pipeline Completed ==========")
        print(f"Final Model Accuracy: {accuracy:.2%}")
        print("====================================================\n")
        
    except Exception as e:
        logging.error(CustomException(e, sys))
        print("\nError occurred during model training. Check the logs for details.\n")
