"""
Test script for the new ML implementations
"""

import json
import requests
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Create a simple test dataset
print("Creating test dataset...")
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=3, 
                          n_classes=2, random_state=42)

# Create a DataFrame
feature_names = [f'feature_{i}' for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
test_file = "datasets/test_ml_training.csv"
df.to_csv(test_file, index=False)
print(f"Test dataset saved to: {test_file}")

# Test the model training endpoint
print("\nTesting model training endpoint...")
training_request = {
    "dataset_path": test_file,
    "target_col": "target",
    "task_type": "classification",
    "models": ["LogisticRegression", "RandomForest"],
    "cv_folds": 3,
    "train_size": 0.8
}

try:
    response = requests.post("http://127.0.0.1:8001/api/model-training", json=training_request, timeout=60)
    if response.status_code == 200:
        result = response.json()
        print("✅ Model training successful!")
        print(f"Models trained: {result['models_trained']}")
        print(f"Best model: {result['best_model']}")
        print(f"Best score: {result['best_score']}")
        print(f"Model paths: {result['model_paths']}")
        
        # Test hyperparameter tuning
        print("\nTesting hyperparameter tuning...")
        tuning_request = {
            "dataset_path": test_file,
            "target_col": "target",
            "model_name": "RandomForest",
            "strategy": "random_search",
            "max_trials": 10
        }
        
        tuning_response = requests.post("http://127.0.0.1:8001/api/hyperparameter-tuning", json=tuning_request, timeout=30)
        if tuning_response.status_code == 200:
            tuning_result = tuning_response.json()
            print("✅ Hyperparameter tuning successful!")
            print(f"Best params: {tuning_result['best_params']}")
            print(f"Best score: {tuning_result['best_score']}")
        else:
            print(f"❌ Hyperparameter tuning failed: {tuning_response.text}")
        
        # Test model evaluation
        print("\nTesting model evaluation...")
        evaluation_request = {
            "dataset_path": test_file,
            "target_col": "target",
            "model_paths": result['model_paths'],
            "metrics": ["accuracy", "precision", "recall", "f1_score"]
        }
        
        eval_response = requests.post("http://127.0.0.1:8001/api/model-evaluation", json=evaluation_request, timeout=30)
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            print("✅ Model evaluation successful!")
            print(f"Models evaluated: {eval_result['models_evaluated']}")
            print(f"Best performing model: {eval_result['best_performing_model']}")
            
            # Test evaluation report generation
            print("\nTesting evaluation report generation...")
            report_request = {
                "evaluation_results": eval_result["evaluation_results"],
                "comparison_results": eval_result["comparison_results"],
                "dataset_name": "Test Dataset"
            }
            
            report_response = requests.post("http://127.0.0.1:8001/api/evaluation-report", json=report_request, timeout=15)
            if report_response.status_code == 200:
                print("✅ Evaluation report generation successful!")
                report_result = report_response.json()
                print(f"Report saved to: {report_result['report_path']}")
            else:
                print(f"❌ Evaluation report generation failed: {report_response.text}")
        else:
            print(f"❌ Model evaluation failed: {eval_response.text}")
        
    else:
        print(f"❌ Model training failed: {response.text}")
        
except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    
print("\nTest complete!")