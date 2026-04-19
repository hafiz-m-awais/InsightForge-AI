"""
Simple test for hyperparameter tuning endpoint
"""

import json
import requests

print("Testing hyperparameter tuning endpoint...")
tuning_request = {
    "dataset_path": "datasets/test_ml_training.csv",
    "target_col": "target",
    "model_name": "RandomForest",
    "strategy": "random_search",
    "max_trials": 5
}

try:
    response = requests.post("http://127.0.0.1:8001/api/hyperparameter-tuning", json=tuning_request, timeout=30)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Response structure:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Error occurred: {str(e)}")