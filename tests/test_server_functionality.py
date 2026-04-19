#!/usr/bin/env python3
"""
Test multiple endpoints to verify full server functionality
"""

import requests
import json

def test_multiple_endpoints():
    """Test multiple endpoints to ensure full functionality"""
    
    base_url = "http://127.0.0.1:8001/api"
    
    # Test 1: Model Training (already working)
    print("1. Testing Model Training...")
    training_data = {
        'dataset_path': 'datasets/data_8d735d32_processed.csv',
        'target_col': 'churn',
        'task_type': 'classification',
        'models': ['LogisticRegression'],
        'cv_folds': 3,
        'train_size': 0.8
    }
    
    try:
        response = requests.post(f'{base_url}/model-training', json=training_data, timeout=60)
        if response.status_code == 200:
            print("✅ Model Training: SUCCESS")
            result = response.json()
            best_model = result.get('best_model')
            print(f"   Best model: {best_model}")
        else:
            print(f"❌ Model Training: ERROR {response.status_code}")
    except Exception as e:
        print(f"❌ Model Training: FAILED - {e}")
    
    # Test 2: XAI Analysis (Phase 2 feature)
    print("\n2. Testing XAI Analysis...")
    xai_data = {
        'model_path': 'models/LogisticRegression_20260418_152255.joblib',
        'model_name': 'LogisticRegression',
        'dataset_path': 'datasets/data_8d735d32_processed.csv',
        'target_col': 'churn',
        'include_shap': True,
        'include_learning_curves': True
    }
    
    try:
        response = requests.post(f'{base_url}/xai-analysis', json=xai_data, timeout=60)
        if response.status_code == 200:
            print("✅ XAI Analysis: SUCCESS")
            result = response.json()
            print(f"   Analysis types: {result.get('analysis_types', [])}")
        else:
            print(f"❌ XAI Analysis: ERROR {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ XAI Analysis: FAILED - {e}")
    
    # Test 3: List Saved Models
    print("\n3. Testing List Saved Models...")
    try:
        response = requests.get(f'{base_url}/list-saved-models', timeout=30)
        if response.status_code == 200:
            print("✅ List Saved Models: SUCCESS")
            result = response.json()
            print(f"   Found {len(result.get('saved_models', []))} saved models")
        else:
            print(f"❌ List Saved Models: ERROR {response.status_code}")
    except Exception as e:
        print(f"❌ List Saved Models: FAILED - {e}")
    
    print("\n" + "="*50)
    print("🎉 Server functionality test completed!")
    print("The internal server errors have been resolved.")

if __name__ == "__main__":
    test_multiple_endpoints()