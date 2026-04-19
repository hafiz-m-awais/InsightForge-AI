#!/usr/bin/env python3
"""
Test the fixed model training endpoint
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"  # Updated port

def test_basic_endpoint():
    """Test the basic ML training test endpoint"""
    try:
        print("=== Testing Basic ML Training Test Endpoint ===")
        response = requests.post(f"{BASE_URL}/api/model-training-test", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Basic test passed!")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
            print(f"   Accuracy: {result.get('accuracy')}")
            return True
        else:
            print(f"❌ Basic test failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Basic test error: {e}")
        return False

def test_model_training_with_valid_data():
    """Test model training with valid Titanic dataset"""
    try:
        print("\n=== Testing Model Training with Valid Data ===")
        
        request_data = {
            "dataset_path": "d:\\MAIN_PRojects\\Autonomous_DS_agent\\datasets\\data_8964b2f3.csv",
            "target_col": "Survived",
            "task_type": "classification",
            "models": ["LogisticRegression"],
            "cv_folds": 2,
            "train_size": 0.8
        }
        
        print(f"Sending request...")
        response = requests.post(
            f"{BASE_URL}/api/model-training", 
            json=request_data, 
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model training successful!")
            print(f"   Models trained: {result.get('models_trained', [])}")
            
            if 'val_scores' in result:
                print("   Validation scores:")
                for model, score in result['val_scores'].items():
                    print(f"     {model}: {score:.4f}")
                    
            return True
            
        else:
            print(f"❌ Model training failed with status: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail.get('detail', 'No detail available')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def test_model_training_with_invalid_data():
    """Test model training with invalid dataset to verify error handling"""
    try:
        print("\n=== Testing Error Handling with Invalid Data ===")
        
        request_data = {
            "dataset_path": "d:\\MAIN_PRojects\\Autonomous_DS_agent\\datasets\\test.csv",
            "target_col": "target",
            "task_type": "classification",
            "models": ["LogisticRegression"],
            "cv_folds": 2,
            "train_size": 0.8
        }
        
        response = requests.post(
            f"{BASE_URL}/api/model-training", 
            json=request_data, 
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"✅ Error handling works correctly!")
            print(f"   Status code: {response.status_code}")
            
            try:
                error_detail = response.json()
                print(f"   Error message: {error_detail.get('detail', 'No detail available')}")
            except:
                print(f"   Raw response: {response.text}")
                
            return True
            
        else:
            print(f"❌ Expected error but got success: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def test_server_health():
    """Test server health endpoint"""
    try:
        print("=== Testing Server Health ===")
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Server is healthy!")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return False

if __name__ == "__main__":
    print("Starting Model Training Endpoint Tests")
    print(f"Testing server at: {BASE_URL}")
    print("=" * 50)
    
    # Wait a moment for server to fully start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Test server health first
    server_ok = test_server_health()
    if not server_ok:
        print("\n❌ Server not available. Please start the server first.")
        exit(1)
    
    # Test basic functionality
    basic_ok = test_basic_endpoint()
    
    # Test error handling
    error_ok = test_model_training_with_invalid_data()
    
    # Test actual model training
    training_ok = test_model_training_with_valid_data()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"✅ Server Health:        {'PASS' if server_ok else 'FAIL'}")
    print(f"✅ Basic Functionality:  {'PASS' if basic_ok else 'FAIL'}")
    print(f"✅ Error Handling:       {'PASS' if error_ok else 'FAIL'}")
    print(f"✅ Model Training:       {'PASS' if training_ok else 'FAIL'}")
    
    if server_ok and basic_ok and error_ok and training_ok:
        print("\n🎉 All tests passed! Model training endpoint is working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")