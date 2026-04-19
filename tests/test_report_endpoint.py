"""
Test just the evaluation report endpoint
"""

import json
import requests

# Sample evaluation results for testing
test_data = {
    "evaluation_results": [
        {
            "model_name": "RandomForest",
            "task_type": "classification",
            "metrics": {
                "accuracy": 0.8700,
                "precision": 0.8737,
                "recall": 0.8700,
                "f1_score": 0.8697,
                "roc_auc": 0.9516,
                "primary_metric": 0.8700
            },
            "feature_importance": [
                {"feature": "feature_7", "importance": 0.1234},
                {"feature": "feature_3", "importance": 0.0987},
                {"feature": "feature_12", "importance": 0.0654},
                {"feature": "feature_1", "importance": 0.0543},
                {"feature": "feature_9", "importance": 0.0432}
            ]
        },
        {
            "model_name": "XGBoost", 
            "task_type": "classification",
            "metrics": {
                "accuracy": 0.8650,
                "precision": 0.8680,
                "recall": 0.8650,
                "f1_score": 0.8640,
                "roc_auc": 0.9480,
                "primary_metric": 0.8650
            },
            "feature_importance": [
                {"feature": "feature_7", "importance": 0.1156},
                {"feature": "feature_3", "importance": 0.0934},
                {"feature": "feature_12", "importance": 0.0678},
                {"feature": "feature_1", "importance": 0.0567},
                {"feature": "feature_9", "importance": 0.0423}
            ]
        }
    ],
    "comparison_results": {
        "best_model": "RandomForest",
        "rankings": [
            {
                "rank": 1,
                "model_name": "RandomForest", 
                "score": 0.8700,
                "metrics": {"accuracy": 0.8700, "precision": 0.8737, "recall": 0.8700, "f1_score": 0.8697}
            },
            {
                "rank": 2,
                "model_name": "XGBoost",
                "score": 0.8650,
                "metrics": {"accuracy": 0.8650, "precision": 0.8680, "recall": 0.8650, "f1_score": 0.8640}
            }
        ],
        "recommendations": [
            "🏆 **RandomForest** is recommended for production deployment.",
            "✅ Good accuracy - suitable for most applications.",
            "📊 Consider XGBoost as alternative - similar performance."
        ]
    },
    "dataset_name": "Test Evaluation Report"
}

try:
    print("Testing evaluation report endpoint...")
    response = requests.post("http://127.0.0.1:8001/api/evaluation-report-real", json=test_data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Evaluation report generation successful!")
        print(f"Report path: {result['report_path']}")
        print(f"HTML length: {len(result['report_html'])} characters")
        print(f"Report preview: {result['report_html'][:200]}...")
    else:
        print(f"❌ Report generation failed: {response.status_code} - {response.text}")

except Exception as e:
    print(f"❌ Error: {str(e)}")