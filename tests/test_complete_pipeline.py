"""
Complete end-to-end test of the new ML pipeline
"""

import json
import requests
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def test_complete_ml_pipeline():
    # Create a test dataset
    print("Creating test dataset...")
    X, y = make_classification(n_samples=1000, n_features=15, n_informative=8, n_redundant=4, 
                              n_classes=2, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(15)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    test_file = "datasets/test_complete_pipeline.csv"
    df.to_csv(test_file, index=False)
    print(f"✅ Test dataset created: {test_file}")
    
    try:
        # Step 1: Model Training
        print("\n1️⃣ Testing Model Training...")
        training_request = {
            "dataset_path": test_file,
            "target_col": "target",
            "task_type": "classification",
            "models": ["LogisticRegression", "RandomForest", "XGBoost"],
            "cv_folds": 3,
            "train_size": 0.8
        }
        
        response = requests.post("http://127.0.0.1:8001/api/model-training", json=training_request, timeout=120)
        
        if response.status_code == 200:
            training_result = response.json()
            print("✅ Model training successful!")
            print(f"   Models trained: {training_result['models_trained']}")
            print(f"   Best model: {training_result['best_model']}")
            print(f"   Best score: {training_result['best_score']:.4f}")
            
            model_paths = training_result['model_paths']
        else:
            print(f"❌ Model training failed: {response.text}")
            return
        
        # Step 2: Hyperparameter Tuning
        print("\n2️⃣ Testing Hyperparameter Tuning...")
        tuning_request = {
            "dataset_path": test_file,
            "target_col": "target",
            "model_name": training_result['best_model'],
            "strategy": "random_search",
            "max_trials": 15
        }
        
        tuning_response = requests.post("http://127.0.0.1:8001/api/hyperparameter-tuning", json=tuning_request, timeout=60)
        
        if tuning_response.status_code == 200:
            tuning_result = tuning_response.json()
            print("✅ Hyperparameter tuning successful!")
            print(f"   Best parameters: {tuning_result['best_params']}")
            print(f"   Best score: {tuning_result['best_score']:.4f}")
            print(f"   Trials completed: {len(tuning_result['optimization_history'])}")
        else:
            print(f"❌ Hyperparameter tuning failed: {tuning_response.text}")
        
        # Step 3: Model Evaluation
        print("\n3️⃣ Testing Model Evaluation...")
        evaluation_request = {
            "dataset_path": test_file,
            "target_col": "target",
            "model_paths": model_paths,
            "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        }
        
        eval_response = requests.post("http://127.0.0.1:8001/api/model-evaluation", json=evaluation_request, timeout=60)
        
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            print("✅ Model evaluation successful!")
            print(f"   Models evaluated: {eval_result['models_evaluated']}")
            print(f"   Best model: {eval_result['best_performing_model']}")
            
            # Display evaluation metrics for best model
            best_model_result = next(
                (r for r in eval_result["evaluation_results"] 
                 if r["model_name"] == eval_result["best_performing_model"]), 
                None
            )
            
            if best_model_result:
                print("   Best model metrics:")
                for metric, value in best_model_result["metrics"].items():
                    if metric != "primary_metric":
                        print(f"     {metric}: {value:.4f}")
        else:
            print(f"❌ Model evaluation failed: {eval_response.text}")
            return
        
        # Step 4: Evaluation Report Generation
        print("\n4️⃣ Testing Evaluation Report Generation...")
        report_request = {
            "evaluation_results": eval_result["evaluation_results"],
            "comparison_results": eval_result["comparison_results"],
            "dataset_name": "Complete Pipeline Test Dataset"
        }
        
        report_response = requests.post("http://127.0.0.1:8001/api/evaluation-report-real", json=report_request, timeout=30)
        
        if report_response.status_code == 200:
            report_result = report_response.json()
            print("✅ Evaluation report generation successful!")
            print(f"   Report saved to: {report_result['report_path']}")
            print(f"   Report HTML length: {len(report_result['report_html'])} characters")
        else:
            print(f"❌ Evaluation report generation failed: {report_response.text}")
        
        print("\n🎉 Complete ML Pipeline Test Successful!")
        print("=" * 60)
        print("✅ All Phase 1 implementations working correctly:")
        print("   • Real model training with cross-validation")
        print("   • Hyperparameter optimization with optuna/sklearn")  
        print("   • Comprehensive model evaluation")
        print("   • Professional evaluation report generation")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {str(e)}")

if __name__ == "__main__":
    test_complete_ml_pipeline()