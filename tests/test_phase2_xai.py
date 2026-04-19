"""
Test script for Phase 2: Advanced XAI Features
Tests SHAP integration, feature importance analysis, and model persistence.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from app.agents.ml_training_agent import MLTrainingAgent
from app.agents.xai_agent import FeatureImportanceAgent, ModelPersistenceAgent, XAIDashboardGenerator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def create_test_dataset():
    """Create a synthetic dataset for testing."""
    print("📊 Creating synthetic test dataset...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save dataset
    dataset_path = Path("datasets/test_xai_dataset.csv")
    dataset_path.parent.mkdir(exist_ok=True)
    df.to_csv(dataset_path, index=False)
    
    print(f"✅ Dataset saved: {dataset_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Target distribution: {np.bincount(y)}")
    
    return str(dataset_path)

def train_test_model(dataset_path: str):
    """Train a test model for XAI analysis."""
    print("\n🎯 Training test model...")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Prepare data
    ml_agent = MLTrainingAgent()
    X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
        df, 'target', test_size=0.2, validation_size=0.15
    )
    
    # Train RandomForest
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = Path("models/test_xai_model.joblib")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"✅ Model trained and saved: {model_path}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    return str(model_path), X_train, X_test, y_train, y_test

def test_feature_importance_analysis(model_path: str, X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """Test comprehensive feature importance analysis."""
    print("\n🔍 Testing Feature Importance Analysis...")
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Initialize feature importance agent
        feature_agent = FeatureImportanceAgent()
        
        # Run comprehensive analysis
        importance_results = feature_agent.calculate_comprehensive_importance(
            model, X_train, X_test, y_test, "TestModel_XAI"
        )
        
        print("✅ Feature importance analysis completed!")
        print(f"   Methods analyzed: {list(importance_results['importance_types'].keys())}")
        
        # Check if SHAP analysis was successful
        if 'shap' in importance_results['importance_types']:
            shap_data = importance_results['importance_types']['shap']
            print(f"   SHAP features analyzed: {len(shap_data.get('feature_importance', []))}")
            print(f"   SHAP plots generated: {len(shap_data.get('plots', {}))}")
        
        # Generate learning curves
        learning_curves = feature_agent.generate_learning_curves(
            model, X_train, y_train, "TestModel_XAI"
        )
        
        if learning_curves:
            print("✅ Learning curves generated!")
        
        return importance_results, learning_curves
        
    except Exception as e:
        print(f"❌ Feature importance analysis failed: {str(e)}")
        return None, None

def test_model_persistence(model_path: str):
    """Test model persistence with metadata."""
    print("\n💾 Testing Model Persistence...")
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Initialize persistence agent
        persistence_agent = ModelPersistenceAgent()
        
        # Prepare metadata
        metadata = {
            "description": "Test model for XAI analysis",
            "algorithm": "RandomForestClassifier",
            "accuracy": "0.85",
            "training_date": "2024-01-15",
            "features_used": "feature_1,feature_2,feature_3,feature_4,feature_5",
            "hyperparameters": '{"n_estimators": 50, "random_state": 42}'
        }
        
        # Save model with metadata
        save_results = persistence_agent.save_model_with_metadata(
            model, "TestModel_XAI_Persistence", metadata
        )
        
        print("✅ Model saved with metadata!")
        print(f"   Model ID: {save_results['model_id']}")
        print(f"   Model path: {save_results['model_path']}")
        print(f"   Metadata path: {save_results['metadata_path']}")
        
        # List saved models
        saved_models = persistence_agent.list_saved_models()
        print(f"   Total saved models: {len(saved_models)}")
        
        return save_results
        
    except Exception as e:
        print(f"❌ Model persistence failed: {str(e)}")
        return None

def test_xai_dashboard(model_results: dict, importance_results: dict):
    """Test XAI dashboard generation."""
    print("\n🎨 Testing XAI Dashboard Generation...")
    
    try:
        # Initialize dashboard generator
        dashboard_generator = XAIDashboardGenerator()
        
        # Mock model results for dashboard
        model_results = {
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.84,
                "recall": 0.86,
                "f1_score": 0.85
            }
        }
        
        # Generate dashboard
        dashboard_path = dashboard_generator.generate_xai_dashboard(
            model_results, importance_results, "TestModel_XAI"
        )
        
        print("✅ XAI Dashboard generated!")
        print(f"   Dashboard path: {dashboard_path}")
        
        return dashboard_path
        
    except Exception as e:
        print(f"❌ Dashboard generation failed: {str(e)}")
        return None

def main():
    """Main test function."""
    print("🚀 Starting Phase 2: Advanced XAI Features Test\n")
    print("=" * 60)
    
    try:
        # Step 1: Create test dataset
        dataset_path = create_test_dataset()
        
        # Step 2: Train test model
        model_path, X_train, X_test, y_train, y_test = train_test_model(dataset_path)
        
        # Step 3: Test feature importance analysis
        importance_results, learning_curves = test_feature_importance_analysis(
            model_path, X_train, X_test, y_train, y_test
        )
        
        # Step 4: Test model persistence
        persistence_results = test_model_persistence(model_path)
        
        # Step 5: Test XAI dashboard generation
        if importance_results:
            dashboard_path = test_xai_dashboard(persistence_results, importance_results)
        
        print("\n" + "=" * 60)
        print("🎉 Phase 2: Advanced XAI Features Test Complete!")
        print("\n📋 Summary:")
        print("✅ Feature Importance Analysis - PASSED" if importance_results else "❌ Feature Importance Analysis - FAILED")
        print("✅ Learning Curves Generation - PASSED" if learning_curves else "❌ Learning Curves Generation - FAILED")
        print("✅ Model Persistence - PASSED" if persistence_results else "❌ Model Persistence - FAILED")
        print("✅ XAI Dashboard - PASSED" if 'dashboard_path' in locals() and dashboard_path else "❌ XAI Dashboard - FAILED")
        
        print("\n🔬 XAI Features Available:")
        if importance_results:
            importance_types = importance_results.get('importance_types', {})
            for method in importance_types.keys():
                print(f"   ✅ {method.replace('_', ' ').title()} Importance")
        
        print("\n📊 Generated Files:")
        print(f"   📁 Dataset: {dataset_path}")
        print(f"   🎯 Model: {model_path}")
        if persistence_results:
            print(f"   💾 Persistent Model: {persistence_results['model_path']}")
            print(f"   📋 Metadata: {persistence_results['metadata_path']}")
        if 'dashboard_path' in locals() and dashboard_path:
            print(f"   🎨 Dashboard: {dashboard_path}")
        
        print("\n🎯 Next Steps:")
        print("   1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("   2. Access XAI Dashboard at: http://localhost:8000 (Step 11)")
        print("   3. Try Model Persistence at: http://localhost:8000 (Step 12)")
        print("   4. Use the test files created above for demo")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()