#!/usr/bin/env python3
"""
Direct test of MLTrainingAgent functionality (no server needed)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_training_agent_directly():
    """Test MLTrainingAgent directly without server"""
    try:
        print("=== Direct MLTrainingAgent Test ===")
        
        # Import the fixed agent
        from app.agents.ml_training_agent import MLTrainingAgent
        import pandas as pd
        
        # Load real dataset
        dataset_path = "datasets/data_8964b2f3.csv"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            return False
            
        print(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if target column exists
        target_col = 'Survived'
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found")
            return False
        
        print(f"Target column '{target_col}' found ✅")
        
        # Create MLTrainingAgent
        print("Creating MLTrainingAgent...")
        agent = MLTrainingAgent()
        
        # Test data preparation
        print("Preparing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = agent.prepare_data(
            df=df, 
            target_column=target_col,
            test_size=0.2,
            validation_size=0.2
        )
        
        print(f"Data prepared successfully:")
        print(f"  Train: {X_train.shape}, {y_train.shape}")
        print(f"  Val:   {X_val.shape}, {y_val.shape}")
        print(f"  Test:  {X_test.shape}, {y_test.shape}")
        
        # Test model training
        print("Training models...")
        results = agent.train_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            models_to_train=['LogisticRegression'],
            cv_folds=2
        )
        
        print(f"✅ Model training completed successfully!")
        print(f"Models trained: {results.get('models_trained', [])}")
        
        if 'val_scores' in results:
            print("Validation scores:")
            for model, score in results['val_scores'].items():
                print(f"  {model}: {score:.4f}")
                
        if 'cv_scores' in results:
            print("Cross-validation scores:")
            for model, scores in results['cv_scores'].items():
                avg_score = sum(scores) / len(scores)
                print(f"  {model}: {avg_score:.4f} (avg of {len(scores)} folds)")
        
        return True
        
    except Exception as e:
        print(f"❌ MLTrainingAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Direct MLTrainingAgent Test")
    print("=" * 50)
    
    success = test_ml_training_agent_directly()
    
    if success:
        print("\n🎉 MLTrainingAgent is working correctly!")
        print("The model training functionality has been fixed successfully.")
    else:
        print("\n❌ MLTrainingAgent test failed.")
        print("Please check the error messages above.")