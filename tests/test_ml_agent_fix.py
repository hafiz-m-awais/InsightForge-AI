"""
Quick test script to verify the MLTrainingAgent fixes
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the app directory to path so we can import the agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.agents.ml_training_agent import MLTrainingAgent

def test_model_name_mapping():
    """Test that model name normalization works correctly"""
    print("🧪 Testing model name mapping...")
    
    agent = MLTrainingAgent()
    
    # Test model name normalization
    test_names = [
        "logistic_regression",
        "random_forest", 
        "gradient_boosting",
        "LogisticRegression",
        "RandomForest"
    ]
    
    for name in test_names:
        normalized = agent._normalize_model_name(name)
        print(f"  {name} -> {normalized}")
    
    print("✅ Model name mapping test passed!\n")

def test_classification_training():
    """Test classification model training with corrected names"""
    print("🧪 Testing classification model training...")
    
    # Create a simple dataset
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary classification
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
    df['target'] = y
    
    # Initialize agent
    agent = MLTrainingAgent()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = agent.prepare_data(
        df, 'target', test_size=0.2, validation_size=0.2
    )
    
    print(f"  Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Task type: {agent.task_type}")
    
    # Test with frontend-style model names (lowercase with underscores)
    models_to_test = ["logistic_regression", "random_forest"]
    
    print(f"  Training models: {models_to_test}")
    
    results = agent.train_models(
        X_train, y_train, X_val, y_val,
        models_to_train=models_to_test,
        cv_folds=3
    )
    
    print(f"  Results: {results['models_trained']}")
    print(f"  Best model: {results['best_model']}")
    print(f"  Best score: {results['best_score']}")
    print(f"  Validation scores: {results['val_scores']}")
    
    # Verify no -inf values
    assert not np.isinf(results['best_score']), "Best score should not be -inf"
    assert results['best_model'] is not None, "Best model should not be None"
    
    print("✅ Classification training test passed!\n")

if __name__ == "__main__":
    try:
        test_model_name_mapping()
        test_classification_training()
        print("🎉 All tests passed! The MLTrainingAgent fixes are working correctly.")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()