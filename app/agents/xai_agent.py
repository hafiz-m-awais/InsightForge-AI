"""
Enhanced Feature Importance and XAI Agent for InsightForge-AI
Provides SHAP-based explanations, permutation importance, and advanced model interpretability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import joblib
import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Sklearn imports for permutation importance
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportanceAgent:
    """Advanced feature importance and explainability agent using SHAP and sklearn."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
    def calculate_comprehensive_importance(self, model, X_train: pd.DataFrame, 
                                        X_test: pd.DataFrame, y_test: pd.Series,
                                        model_name: str = "Model") -> Dict[str, Any]:
        """Calculate multiple types of feature importance."""
        logger.info(f"Calculating comprehensive feature importance for {model_name}")
        
        importance_results = {
            "model_name": model_name,
            "feature_names": X_test.columns.tolist(),
            "importance_types": {}
        }
        
        # 1. Built-in feature importance (tree-based models)
        if hasattr(model, 'feature_importances_'):
            tree_importance = self._get_tree_importance(model, X_test.columns)
            importance_results["importance_types"]["tree_based"] = tree_importance
            
        # 2. Permutation importance
        perm_importance = self._get_permutation_importance(model, X_test, y_test, X_test.columns)
        importance_results["importance_types"]["permutation"] = perm_importance
        
        # 3. SHAP importance (if available)
        if SHAP_AVAILABLE:
            shap_importance = self._get_shap_importance(model, X_train, X_test, model_name)
            if shap_importance:
                importance_results["importance_types"]["shap"] = shap_importance
        
        # 4. Generate visualizations
        visualizations = self._generate_importance_plots(importance_results, model_name)
        importance_results["visualizations"] = visualizations
        
        return importance_results
    
    def _get_tree_importance(self, model, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Get tree-based feature importance."""
        importance = model.feature_importances_
        
        feature_importance = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, importance)
        ]
        
        return sorted(feature_importance, key=lambda x: x["importance"], reverse=True)
    
    def _get_permutation_importance(self, model, X_test: pd.DataFrame, 
                                   y_test: pd.Series, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Calculate permutation importance."""
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=self.random_state
            )
            
            feature_importance = [
                {
                    "feature": name, 
                    "importance": float(perm_importance.importances_mean[i]),
                    "std": float(perm_importance.importances_std[i])
                }
                for i, name in enumerate(feature_names)
            ]
            
            return sorted(feature_importance, key=lambda x: x["importance"], reverse=True)
            
        except Exception as e:
            logger.warning(f"Permutation importance calculation failed: {str(e)}")
            return []
    
    def _get_shap_importance(self, model, X_train: pd.DataFrame, 
                           X_test: pd.DataFrame, model_name: str) -> Optional[Dict[str, Any]]:
        """Calculate SHAP-based importance and explanations."""
        try:
            # Choose appropriate SHAP explainer
            explainer = None
            
            # Tree explainer for tree-based models
            if any(model_type in str(type(model)) for model_type in ['RandomForest', 'GradientBoosting', 'XGB']):
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {str(e)}, trying alternative...")
                    explainer = None
                    
            # Linear explainer for linear models
            elif any(model_type in str(type(model)) for model_type in ['Linear', 'Logistic']):
                try:
                    explainer = shap.LinearExplainer(model, X_train)
                except Exception as e:
                    logger.warning(f"LinearExplainer failed: {str(e)}, trying alternative...")
                    explainer = None
            
            # Kernel explainer as fallback (slower but works for any model)
            if explainer is None:
                # Use a sample for kernel explainer to avoid memory issues
                sample_size = min(50, len(X_train))  # Smaller sample for kernel explainer
                background = X_train.sample(sample_size, random_state=self.random_state)
                explainer = shap.KernelExplainer(model.predict_proba, background)
            
            # Calculate SHAP values for test set (limit sample size for performance)
            test_sample_size = min(50, len(X_test))  # Smaller sample for stability
            X_test_sample = X_test.sample(test_sample_size, random_state=self.random_state)
            
            shap_values = explainer.shap_values(X_test_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification - use first class or class 1 if binary
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # Use positive class for binary classification
                else:
                    shap_values = shap_values[0]  # Use first class for multiclass
            
            # Ensure shap_values is 2D
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Validate shapes
            if shap_values.shape[1] != len(X_test_sample.columns):
                logger.warning(f"SHAP values shape mismatch: {shap_values.shape[1]} vs {len(X_test_sample.columns)}")
                return None
            
            # Calculate feature importance as mean absolute SHAP values
            mean_shap_importance = np.abs(shap_values).mean(axis=0)
            
            feature_importance = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(X_test_sample.columns, mean_shap_importance)
            ]
            
            # Generate SHAP plots (skip if problematic)
            shap_plots = {}
            try:
                shap_plots = self._generate_shap_plots(
                    explainer, shap_values, X_test_sample, model_name
                )
            except Exception as plot_error:
                logger.warning(f"SHAP plot generation failed: {str(plot_error)}")
            
            return {
                "feature_importance": sorted(feature_importance, key=lambda x: x["importance"], reverse=True),
                "plots": shap_plots,
                "sample_size": test_sample_size,
                "explainer_type": type(explainer).__name__
            }
            
        except Exception as e:
            logger.warning(f"SHAP importance calculation failed: {str(e)}")
            return None
    
    def _generate_shap_plots(self, explainer, shap_values: np.ndarray, 
                           X_sample: pd.DataFrame, model_name: str) -> Dict[str, str]:
        """Generate SHAP visualization plots."""
        plots = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            summary_plot_path = self.plots_dir / f"shap_summary_{model_name}_{timestamp}.png"
            plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots["summary"] = str(summary_plot_path)
            
            # 2. Feature importance bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            bar_plot_path = self.plots_dir / f"shap_bar_{model_name}_{timestamp}.png"
            plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots["bar"] = str(bar_plot_path)
            
            # 3. Waterfall plot for first instance (only for SHAP v0.40+)
            if len(shap_values) > 0 and hasattr(shap, 'waterfall_plot'):
                try:
                    plt.figure(figsize=(10, 6))
                    # Use newer SHAP waterfall plot if available
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[0],
                            base_values=explainer.expected_value,
                            data=X_sample.iloc[0]
                        ),
                        show=False
                    )
                    waterfall_plot_path = self.plots_dir / f"shap_waterfall_{model_name}_{timestamp}.png"
                    plt.savefig(waterfall_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    plots["waterfall"] = str(waterfall_plot_path)
                except Exception as waterfall_error:
                    logger.warning(f"Waterfall plot generation failed: {str(waterfall_error)}")
                
        except Exception as e:
            logger.warning(f"SHAP plot generation failed: {str(e)}")
        
        return plots
    
    def _generate_importance_plots(self, importance_results: Dict, model_name: str) -> Dict[str, str]:
        """Generate feature importance comparison plots."""
        plots = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Comparison plot of different importance methods
            fig, axes = plt.subplots(1, len(importance_results["importance_types"]), 
                                   figsize=(5 * len(importance_results["importance_types"]), 6))
            
            if len(importance_results["importance_types"]) == 1:
                axes = [axes]
            
            for idx, (method, importance_data) in enumerate(importance_results["importance_types"].items()):
                if method == "shap":
                    importance_data = importance_data["feature_importance"]
                
                # Take top 10 features
                top_features = importance_data[:10]
                features = [item["feature"] for item in top_features]
                importances = [item["importance"] for item in top_features]
                
                axes[idx].barh(range(len(features)), importances)
                axes[idx].set_yticks(range(len(features)))
                axes[idx].set_yticklabels(features)
                axes[idx].set_title(f"{method.replace('_', ' ').title()} Importance")
                axes[idx].invert_yaxis()
            
            plt.tight_layout()
            comparison_plot_path = self.plots_dir / f"importance_comparison_{model_name}_{timestamp}.png"
            plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots["comparison"] = str(comparison_plot_path)
            
        except Exception as e:
            logger.warning(f"Importance plot generation failed: {str(e)}")
        
        return plots
    
    def generate_learning_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                               model_name: str, cv: int = 5) -> Dict[str, str]:
        """Generate learning curves to assess model performance vs training size."""
        logger.info(f"Generating learning curves for {model_name}")
        
        try:
            # Calculate learning curves
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=cv, train_sizes=train_sizes,
                random_state=self.random_state, n_jobs=-1
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score', color='blue')
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes_abs, val_mean, 'o-', label='Cross-validation score', color='red')
            plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.plots_dir / f"learning_curves_{model_name}_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return {
                "learning_curves": str(plot_path),
                "train_scores": train_scores.tolist(),
                "val_scores": val_scores.tolist(),
                "train_sizes": train_sizes_abs.tolist()
            }
            
        except Exception as e:
            logger.error(f"Learning curves generation failed: {str(e)}")
            return {}
    
    def generate_enhanced_confusion_matrix(self, y_true, y_pred, model_name: str, 
                                         class_names: Optional[List[str]] = None) -> str:
        """Generate an enhanced confusion matrix visualization."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names or range(len(cm)),
                       yticklabels=class_names or range(len(cm)))
            
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.plots_dir / f"confusion_matrix_{model_name}_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Confusion matrix generation failed: {str(e)}")
            return ""
    
    def generate_roc_pr_curves(self, y_true, y_pred_proba, model_name: str) -> Dict[str, str]:
        """Generate ROC and Precision-Recall curves."""
        plots = {}
        
        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            
            plt.figure(figsize=(12, 5))
            
            # ROC subplot
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f'{model_name}', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Precision-Recall subplot
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.plots_dir / f"roc_pr_curves_{model_name}_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plots["roc_pr_curves"] = str(plot_path)
            
            return plots
            
        except Exception as e:
            logger.error(f"ROC/PR curves generation failed: {str(e)}")
            return {}
    
    def plot_to_base64(self, plot_path: str) -> str:
        """Convert plot to base64 string for embedding in HTML."""
        try:
            with open(plot_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Plot to base64 conversion failed: {str(e)}")
            return ""


class ModelPersistenceAgent:
    """Handles model persistence, versioning, and metadata tracking."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.metadata_dir = Path("models/metadata")
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
    def save_model_with_metadata(self, model, model_name: str, 
                                metadata: Dict[str, Any]) -> Dict[str, str]:
        """Save model with comprehensive metadata tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = self.models_dir / model_filename
        joblib.dump(model, model_path)
        
        # Enhance metadata
        enhanced_metadata = {
            "model_name": model_name,
            "model_path": str(model_path),
            "timestamp": timestamp,
            "model_type": str(type(model).__name__),
            "sklearn_version": self._get_sklearn_version(),
            "file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
            **metadata
        }
        
        # Save metadata
        metadata_filename = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "model_id": f"{model_name}_{timestamp}"
        }
    
    def load_model_with_metadata(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and its metadata."""
        model_path = self.models_dir / f"{model_id}.joblib"
        metadata_path = self.metadata_dir / f"{model_id}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """List all saved models with their metadata."""
        models = []
        
        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)
            except Exception as e:
                logger.warning(f"Could not read metadata file {metadata_file}: {str(e)}")
        
        return sorted(models, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version."""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"


class XAIDashboardGenerator:
    """Generates comprehensive explainability dashboards."""
    
    def __init__(self):
        self.dashboard_dir = Path("dashboards")
        self.dashboard_dir.mkdir(exist_ok=True)
        
    def generate_xai_dashboard(self, model_results: Dict[str, Any], 
                             importance_results: Dict[str, Any],
                             model_name: str) -> str:
        """Generate a comprehensive XAI dashboard in HTML format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_filename = f"xai_dashboard_{model_name}_{timestamp}.html"
        dashboard_path = self.dashboard_dir / dashboard_filename
        
        # Generate HTML content
        html_content = self._build_dashboard_html(model_results, importance_results, model_name)
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"XAI Dashboard generated: {dashboard_path}")
        return str(dashboard_path)
    
    def _build_dashboard_html(self, model_results: Dict[str, Any], 
                            importance_results: Dict[str, Any], model_name: str) -> str:
        """Build comprehensive XAI dashboard HTML."""
        html_parts = []
        
        # HTML structure with enhanced styling
        html_parts.extend([
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>XAI Dashboard - {model_name}</title>",
            "<style>",
            self._get_dashboard_css(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='dashboard-container'>",
            f"<h1>🔍 XAI Dashboard - {model_name}</h1>",
            f"<p class='subtitle'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ])
        
        # Model Performance Section
        html_parts.extend([
            "<div class='section'>",
            "<h2>📊 Model Performance</h2>",
            "<div class='metrics-grid'>",
        ])
        
        for metric, value in model_results.get("metrics", {}).items():
            if metric != "primary_metric":
                html_parts.extend([
                    "<div class='metric-card'>",
                    f"<div class='metric-value'>{value:.4f}</div>",
                    f"<div class='metric-label'>{metric.replace('_', ' ').title()}</div>",
                    "</div>",
                ])
        
        html_parts.extend(["</div>", "</div>"])
        
        # Feature Importance Section
        if importance_results.get("importance_types"):
            html_parts.extend([
                "<div class='section'>",
                "<h2>🎯 Feature Importance Analysis</h2>",
            ])
            
            for method, importance_data in importance_results["importance_types"].items():
                html_parts.extend([
                    f"<h3>{method.replace('_', ' ').title()} Importance</h3>",
                    "<div class='importance-list'>",
                ])
                
                # Handle different data structures
                if method == "shap":
                    features = importance_data.get("feature_importance", [])[:10]
                else:
                    features = importance_data[:10]
                
                for feature in features:
                    importance_val = feature.get("importance", 0)
                    html_parts.extend([
                        "<div class='feature-item'>",
                        f"<span class='feature-name'>{feature['feature']}</span>",
                        f"<span class='feature-value'>{importance_val:.4f}</span>",
                        "</div>",
                    ])
                
                html_parts.extend(["</div>"])
            
            html_parts.extend(["</div>"])
        
        # SHAP Visualizations Section
        shap_data = importance_results.get("importance_types", {}).get("shap", {})
        if shap_data.get("plots"):
            html_parts.extend([
                "<div class='section'>",
                "<h2>🔬 SHAP Explanations</h2>",
                "<div class='plots-grid'>",
            ])
            
            for plot_type, plot_path in shap_data["plots"].items():
                if os.path.exists(plot_path):
                    html_parts.extend([
                        "<div class='plot-container'>",
                        f"<h4>{plot_type.replace('_', ' ').title()} Plot</h4>",
                        f"<img src='{plot_path}' alt='{plot_type} plot' class='plot-image'>",
                        "</div>",
                    ])
            
            html_parts.extend(["</div>", "</div>"])
        
        # Close HTML
        html_parts.extend([
            "</div>",
            "</body>",
            "</html>",
        ])
        
        return "\n".join(html_parts)
    
    def _get_dashboard_css(self) -> str:
        """Get CSS styling for the XAI dashboard."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            line-height: 1.6;
        }
        
        .dashboard-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 30px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 30px;
        }
        
        .section {
            margin-bottom: 40px;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #7f8c8d;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }
        
        .importance-list {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .feature-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .feature-item:last-child {
            border-bottom: none;
        }
        
        .feature-name {
            font-weight: 500;
            color: #2c3e50;
        }
        
        .feature-value {
            font-weight: bold;
            color: #27ae60;
            background: #e8f5e8;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .plot-container h4 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .plot-image {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        """