#!/usr/bin/env python3
"""
Isolated test of ML training functionality to identify issues
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_sklearn():
    """Test basic sklearn functionality"""
    try:
        print("=== Testing Basic Sklearn Functionality ===")
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test LogisticRegression
        print("Testing LogisticRegression...")
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"  LogisticRegression Accuracy: {lr_acc:.4f}")
        
        # Test RandomForest
        print("Testing RandomForestClassifier...")
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"  RandomForest Accuracy: {rf_acc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Basic sklearn test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_training_agent():
    """Test MLTrainingAgent with real data"""
    try:
        print("\n=== Testing MLTrainingAgent ===")
        
        # Add project root to path
        sys.path.append('d:\\MAIN_PRojects\\Autonomous_DS_agent')
        
        from app.agents.ml_training_agent import MLTrainingAgent
        
        print("MLTrainingAgent imported successfully")
        
        # Load real dataset
        dataset_path = "d:\\MAIN_PRojects\\Autonomous_DS_agent\\datasets\\data_8964b2f3.csv"
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return False
            
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if target column exists
        if 'Survived' not in df.columns:
            print("Target column 'Survived' not found")
            return False
        
        print("Creating MLTrainingAgent...")
        agent = MLTrainingAgent()
        
        print("Testing data preparation...")
        X_train, X_val, X_test, y_train, y_val, y_test = agent.prepare_data(df, 'Survived', 0.2, 0.2)
        print(f"Data prepared: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # Test model training with limited models to avoid issues
        print("Testing model training...")
        request_data = {
            'dataset_path': dataset_path,
            'target_col': 'Survived',
            'task_type': 'classification',
            'models': ['LogisticRegression'],  # Just one model for testing
            'cv_folds': 2,  # Reduce CV folds
            'train_size': 0.8
        }
        
        results = agent.train_models(
            dataset_path=request_data['dataset_path'],
            target_col=request_data['target_col'],
            task_type=request_data['task_type'],
            models=request_data['models'],
            cv_folds=request_data['cv_folds'],
            train_size=request_data['train_size']
        )
        
        print(f"Training completed successfully!")
        print(f"Results keys: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"MLTrainingAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_titanic_data():
    """Test with Titanic dataset specifically"""
    try:
        print("\n=== Testing with Titanic Dataset ===")
        
        dataset_path = "d:\\MAIN_PRojects\\Autonomous_DS_agent\\datasets\\data_8964b2f3.csv"
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return False
            
        df = pd.read_csv(dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Target column 'Survived' value counts:")
        if 'Survived' in df.columns:
            print(df['Survived'].value_counts())
        else:
            print("No 'Survived' column found!")
            return False
            
        # Check for missing values
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        # Simple manual training test
        print("\nTesting manual model training...")
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score
        
        # Select numeric columns only for simple test
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Survived' in numeric_cols:
            numeric_cols.remove('Survived')
        
        print(f"Using numeric columns: {numeric_cols}")
        
        X = df[numeric_cols].fillna(0)  # Simple imputation
        y = df['Survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Manual training successful! Accuracy: {accuracy:.4f}")
        return True
        
    except Exception as e:
        print(f"Titanic dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting ML Training Diagnostic Tests")
    print("=" * 50)
    
    # Test 1: Basic sklearn functionality
    sklearn_ok = test_basic_sklearn()
    
    # Test 2: Titanic dataset manual training
    titanic_ok = test_with_titanic_data()
    
    # Test 3: MLTrainingAgent (might fail due to dependencies)
    ml_agent_ok = test_ml_training_agent()
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"✅ Basic sklearn:        {'PASS' if sklearn_ok else 'FAIL'}")
    print(f"✅ Titanic manual:       {'PASS' if titanic_ok else 'FAIL'}")
    print(f"✅ MLTrainingAgent:      {'PASS' if ml_agent_ok else 'FAIL'}")
    
    if sklearn_ok and titanic_ok:
        print("\n✅ Core ML functionality is working!")
        if not ml_agent_ok:
            print("⚠️  Issue is likely in MLTrainingAgent dependencies or implementation")
    else:
        print("\n❌ Basic ML functionality has issues")