"""
Comprehensive ML Training Agent for InsightForge-AI
Handles model training, cross-validation, hyperparameter optimization, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import joblib
from pathlib import Path

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)

# Model imports
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# XGBoost
try:
    import xgboost as _xgb  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Hyperparameter optimization
import importlib.util as _importlib_util
OPTUNA_AVAILABLE = _importlib_util.find_spec("optuna") is not None  # type: ignore[assignment]
if not OPTUNA_AVAILABLE:
    logging.warning("Optuna not available. Install with: pip install optuna")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLTrainingAgent:
    """Comprehensive ML training agent with support for classification and regression."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Preprocessing artifacts
        self.categorical_encoders = {}
        self.imputation_values = {}
        self.categorical_columns_seen = []
        self.numeric_columns_seen = []
        self.label_encoder = None
        
        # Classification models
        self.classification_models = {
            "LogisticRegression": LogisticRegression(random_state=random_state, max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=random_state, n_estimators=100),
            "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
            "SVM": SVC(random_state=random_state, probability=True),
            "NeuralNetwork": MLPClassifier(random_state=random_state, max_iter=500),
            "NaiveBayes": GaussianNB(),
            "KNN": KNeighborsClassifier()
        }
        
        # Regression models
        self.regression_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state),
            "RandomForest": RandomForestRegressor(random_state=random_state, n_estimators=100),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
            "SVR": SVR(),
            "NeuralNetwork": MLPRegressor(random_state=random_state, max_iter=500),
            "KNN": KNeighborsRegressor()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            from xgboost import XGBClassifier, XGBRegressor  # type: ignore[import]
            self.classification_models["XGBoost"] = XGBClassifier(random_state=random_state)
            self.regression_models["XGBoost"] = XGBRegressor(random_state=random_state)
        
        # Model name mapping for different naming conventions
        self.model_name_mapping = {
            # Classification models - support both camelCase and snake_case
            "logistic_regression": "LogisticRegression",
            "logisticregression": "LogisticRegression",
            "random_forest": "RandomForest", 
            "randomforest": "RandomForest",
            "gradient_boosting": "GradientBoosting",
            "gradientboosting": "GradientBoosting", 
            "xgboost": "XGBoost",
            "svm": "SVM",
            "neural_network": "NeuralNetwork",
            "neuralnetwork": "NeuralNetwork",
            "naive_bayes": "NaiveBayes",
            "naivebayes": "NaiveBayes", 
            "knn": "KNN",
            
            # Regression models
            "linear_regression": "LinearRegression",
            "linearregression": "LinearRegression", 
            "ridge": "Ridge",
            "lasso": "Lasso",
            "svr": "SVR",
            
            # Direct mappings (already correct format)
            "LogisticRegression": "LogisticRegression",
            "RandomForest": "RandomForest",
            "GradientBoosting": "GradientBoosting",
            "XGBoost": "XGBoost",
            "SVM": "SVM",
            "NeuralNetwork": "NeuralNetwork",
            "NaiveBayes": "NaiveBayes",
            "KNN": "KNN",
            "LinearRegression": "LinearRegression",
            "Ridge": "Ridge", 
            "Lasso": "Lasso",
            "SVR": "SVR"
        }
        
        # Hyperparameter grids
        self.param_grids = self._get_parameter_grids()
        
    def _get_parameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for optimization."""
        return {
            "LogisticRegression": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            },
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "GradientBoosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            },
            "SVM": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            },
            "NeuralNetwork": {
                "hidden_layer_sizes": [(50,), (100,), (50, 30), (100, 50)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0]
            } if XGBOOST_AVAILABLE else {}
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2, validation_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare data with train/validation/test splits and basic preprocessing."""
        logger.info(f"Preparing data with target column: {target_column}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Determine if it's a classification or regression task
        self.task_type = self._determine_task_type(y)
        logger.info(f"Detected task type: {self.task_type}")
        
        # For classification, encode target if needed
        self.label_encoder = None
        if self.task_type == "classification" and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = pd.Series(np.array(self.label_encoder.fit_transform(y)), name=y.name)
        
        # First split: train+val vs test
        stratify = y if self.task_type == "classification" else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=stratify
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        stratify_temp = y_temp if self.task_type == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=stratify_temp
        )
        
        # Basic feature preprocessing
        X_train = self._preprocess_features(X_train, fit=True)
        X_val = self._preprocess_features(X_val, fit=False)
        X_test = self._preprocess_features(X_test, fit=False)
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Basic feature preprocessing for ML models."""
        X = X.copy()
        
        # Drop columns that are likely pure identifiers: 100% unique values
        # and either integer or string type (heuristic, avoids Titanic-specific hardcoding)
        n = len(X)
        id_columns = [
            col for col in X.columns
            if n > 0 and X[col].nunique() == n
            and (
                pd.api.types.is_integer_dtype(X[col])
                or pd.api.types.is_object_dtype(X[col])
            )
        ]
        for col in id_columns:
            if col in X.columns:
                X = X.drop(columns=[col])
                logger.info(f"Dropped identifier column: {col}")
        
        if fit:
            self.numeric_columns_seen = list(X.select_dtypes(include=[np.number]).columns)
            self.categorical_columns_seen = list(X.select_dtypes(include=['object']).columns)
            self.imputation_values = {}
            self.categorical_encoders = {}
            
            # Numeric imputation
            for col in self.numeric_columns_seen:
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback for completely empty columns
                self.imputation_values[col] = median_val
                if X[col].isnull().any():
                    X[col] = X[col].fillna(median_val)
            
            # Categorical imputation and encoding
            for col in self.categorical_columns_seen:
                mode_s = X[col].mode()
                mode_val = mode_s.iloc[0] if not mode_s.empty else 'Unknown'
                self.imputation_values[col] = mode_val
                
                if X[col].isnull().any():
                    X[col] = X[col].fillna(mode_val)
                
                encoder = LabelEncoder()
                # Create 'Unknown' category placeholder by keeping the string type handling robust
                X[col] = pd.Series(np.array(encoder.fit_transform(X[col].astype(str))), index=X.index, dtype=int)
                
                # Add 'Unknown' fallback handling to internal classes
                classes = list(encoder.classes_)
                if 'Unknown' not in classes:
                    classes.append('Unknown')
                encoder.classes_ = np.array(classes)
                
                self.categorical_encoders[col] = encoder
                logger.info(f"Encoded categorical column: {col}")
                
        else:
            # Apply seen transformations for numeric
            for col in self.numeric_columns_seen:
                if col in X.columns:
                    X[col] = X[col].fillna(self.imputation_values.get(col, 0))
            
            # Apply seen transformations for categorical
            for col in self.categorical_columns_seen:
                if col in X.columns:
                    X[col] = X[col].fillna(self.imputation_values.get(col, 'Unknown'))
                    
                    if col in self.categorical_encoders:
                        encoder = self.categorical_encoders[col]
                        # Handle unseen categories by mapping them to 'Unknown'
                        X_series = X[col].astype(str)
                        known_classes = set(encoder.classes_)
                        X_series = X_series.map(lambda s: s if s in known_classes else 'Unknown')
                        
                        try:
                            X[col] = pd.Series(encoder.transform(X_series), index=X.index).astype(int)
                        except ValueError as e:
                            logger.warning(f"Unknown category handling failed in {col}: {e}. Fallback to 0.")
                            X[col] = 0
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric, if fails encode
                try:
                    X[col] = pd.to_numeric(X[col])
                except Exception:
                    logger.warning(f"Could not convert {col} to numeric")
                    X[col] = 0
        
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        
        return X
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if the task is classification or regression."""
        if y.dtype == 'object' or len(y.unique()) <= 20:
            return "classification"
        else:
            return "regression"
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to standard format."""
        normalized = self.model_name_mapping.get(model_name.lower(), model_name)
        if normalized != model_name and normalized in self.model_name_mapping.values():
            logger.info(f"Mapped model name: {model_name} -> {normalized}")
        return normalized
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    models_to_train: Optional[List[str]] = None,
                    cv_folds: int = 5) -> Dict[str, Any]:
        """Train multiple models with cross-validation."""
        if models_to_train is None:
            if self.task_type == "classification":
                models_to_train = list(self.classification_models.keys())
            else:
                models_to_train = list(self.regression_models.keys())
        
        # Normalize model names
        normalized_models = []
        for model_name in models_to_train:
            normalized_name = self._normalize_model_name(model_name)
            normalized_models.append(normalized_name)
        
        models_to_train = normalized_models
        logger.info(f"Training {len(models_to_train)} models for {self.task_type}")
        
        results = {
            "models_trained": models_to_train,
            "cv_scores": {},
            "val_scores": {},
            "training_times": {},
            "model_paths": {},
            "preprocessor_paths": {},
            "task_type": self.task_type
        }
        
        # Choose appropriate models and scoring
        model_dict = (self.classification_models if self.task_type == "classification" 
                     else self.regression_models)
        scoring = self._get_scoring_metric()
        
        # Setup cross-validation
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        best_score = -np.inf if self.task_type == "classification" else np.inf
        best_model_name = None
        
        for model_name in models_to_train:
            if model_name not in model_dict:
                logger.warning(f"Model {model_name} not available for {self.task_type}")
                continue
                
            start_time = datetime.now()
            
            try:
                model = model_dict[model_name]
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
                results["cv_scores"][model_name] = cv_scores.tolist()
                
                # Train on full training set and validate
                model.fit(X_train, y_train)
                val_predictions = model.predict(X_val)
                val_score = self._calculate_score(y_val, val_predictions)
                results["val_scores"][model_name] = val_score
                
                # Save model
                model_path = self.models_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                joblib.dump(model, model_path)
                results["model_paths"][model_name] = str(model_path)

                # ── Save companion preprocessor ──────────────────────────────
                # Capture the exact transforms applied during training so the
                # prediction endpoint can reproduce the same pipeline at inference.
                preprocessor_path = model_path.with_name(model_path.stem + "_preprocessor.joblib")

                # Try to load FE-level transforms (label encoders, scaler) saved
                # by feature_engineer.py next to the engineered CSV.
                fe_transforms: dict = {}
                _fe_path = getattr(self, "_fe_transforms_path", None)
                if _fe_path:
                    _fe_p = Path(_fe_path)
                    if _fe_p.exists():
                        try:
                            fe_transforms = joblib.load(_fe_p)
                            logger.info("Loaded FE transforms from %s", _fe_p)
                        except Exception as _exc:
                            logger.warning("Could not load FE transforms: %s", _exc)

                preprocessor_data = {
                    "numeric_columns_seen":    list(self.numeric_columns_seen),
                    "categorical_columns_seen": list(self.categorical_columns_seen),
                    "imputation_values": {
                        k: (v.item() if hasattr(v, "item") else v)
                        for k, v in self.imputation_values.items()
                    },
                    "categorical_encoders": dict(self.categorical_encoders),
                    "task_type":       getattr(self, "task_type", "classification"),
                    "label_encoder":   self.label_encoder,
                    "feature_order":   list(X_train.columns),
                    "fe_transforms":   fe_transforms,  # FE-level label encoders + scaler
                    # Per-feature statistics for the training set (numeric columns only)
                    "feature_stats": {
                        col: {
                            "min":  float(X_train[col].min()),
                            "max":  float(X_train[col].max()),
                            "mean": float(X_train[col].mean()),
                        }
                        for col in self.numeric_columns_seen
                        if col in X_train.columns
                    },
                }
                joblib.dump(preprocessor_data, preprocessor_path)
                results["preprocessor_paths"][model_name] = str(preprocessor_path)
                logger.info("Saved preprocessor → %s", preprocessor_path)

                # Track training time
                training_time = (datetime.now() - start_time).total_seconds()
                results["training_times"][model_name] = round(training_time, 2)
                
                # Update best model
                if self._is_better_score(val_score, best_score):
                    best_score = val_score
                    best_model_name = model_name
                
                logger.info(f"{model_name}: CV={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}, "
                           f"Val={val_score:.4f}, Time={training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results["cv_scores"][model_name] = []
                results["val_scores"][model_name] = 0.0
        
        results["best_model"] = best_model_name
        # Handle -inf values that are not JSON serializable
        if best_model_name is None or np.isinf(best_score):
            results["best_score"] = 0.0
            logger.warning("No models were successfully trained. Setting best_score to 0.0")
        else:
            results["best_score"] = best_score
        
        logger.info(f"Training complete. Best model: {best_model_name} (score: {results['best_score']:.4f})")
        return results
    
    def _get_scoring_metric(self) -> str:
        """Get appropriate scoring metric for the task."""
        return "accuracy" if self.task_type == "classification" else "neg_mean_squared_error"
    
    def _calculate_score(self, y_true, y_pred) -> float:
        """Calculate validation score."""
        if self.task_type == "classification":
            return float(accuracy_score(y_true, y_pred))
        else:
            return -mean_squared_error(y_true, y_pred)  # Negative MSE for consistency
    
    def _is_better_score(self, new_score: float, best_score: float) -> bool:
        """Check if new score is better than current best."""
        return new_score > best_score
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                model_name: str, strategy: str = "random_search",
                                max_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model."""
        logger.info(f"Optimizing hyperparameters for {model_name} using {strategy}")
        
        if strategy == "bayesian" and not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to random_search")
            strategy = "random_search"
        model_dict = (self.classification_models if self.task_type == "classification" 
                     else self.regression_models)
        
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not available for {self.task_type}")
        
        base_model = model_dict[model_name]
        param_grid = self.param_grids.get(model_name, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {model_name}")
            return {"best_params": {}, "best_score": 0.0, "optimization_history": []}
        
        if strategy == "bayesian" and OPTUNA_AVAILABLE:
            return self._bayesian_optimization(base_model, param_grid, X_train, y_train, max_trials)
        else:
            return self._grid_random_search(base_model, param_grid, X_train, y_train, strategy, max_trials)
    
    def _bayesian_optimization(self, base_model, param_grid: Dict, X_train: pd.DataFrame, 
                              y_train: pd.Series, max_trials: int) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna."""
        import optuna  # type: ignore  # noqa: PLC0415
        def objective(trial):
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], (int, float)):
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            model = base_model.__class__(**params, random_state=self.random_state)
            scoring = self._get_scoring_metric()
            
            if self.task_type == "classification":
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
        
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "optimization_history": [
                {"trial": i, "score": trial.value, "params": trial.params}
                for i, trial in enumerate(study.trials)
            ]
        }
    
    def _grid_random_search(self, base_model, param_grid: Dict, X_train: pd.DataFrame,
                           y_train: pd.Series, strategy: str, max_trials: int) -> Dict[str, Any]:
        """Perform grid or random search."""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        scoring = self._get_scoring_metric()
        
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        if strategy == "grid_search":
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        else:  # random_search
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv, scoring=scoring, 
                n_iter=max_trials, random_state=self.random_state, n_jobs=-1
            )
        
        search.fit(X_train, y_train)
        
        # Extract optimization history
        history = []
        for i, (params, score) in enumerate(zip(search.cv_results_['params'], search.cv_results_['mean_test_score'])):
            history.append({"trial": i, "score": score, "params": params})
        
        return {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "optimization_history": history
        }
    
    def evaluate_model(self, model_path: str, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: Optional[str] = None, X_train: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation with XAI features."""
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba") and self.task_type == "classification":
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None
        
        # Calculate metrics
        if self.task_type == "classification":
            metrics = self._calculate_classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Basic feature importance
        feature_importance = self._get_feature_importance(model, list(X_test.columns))
        
        evaluation_result = {
            "model_name": model_name or "Unknown",
            "task_type": self.task_type,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "predictions": y_pred.tolist() if len(y_pred) <= 1000 else y_pred[:1000].tolist(),
            "test_size": len(y_test)
        }
        
        # Add classification-specific results
        if self.task_type == "classification":
            evaluation_result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            evaluation_result["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
            
            if y_prob is not None and y_prob.shape[1] == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                evaluation_result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                evaluation_result["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
        
        logger.info(f"Evaluation complete. Primary metric: {metrics.get('primary_metric', 'N/A')}")
        return evaluation_result
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        
        metrics["primary_metric"] = metrics["accuracy"]
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred)
        }
        
        metrics["primary_metric"] = metrics["r2_score"]
        return metrics
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extract feature importance from model."""
        importance = None
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        
        # Linear models
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_)
        
        if importance is not None:
            feature_importance = [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(feature_names, importance)
            ]
            return sorted(feature_importance, key=lambda x: x["importance"], reverse=True)
        
        return []
    
    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model evaluation results."""
        logger.info(f"Comparing {len(evaluation_results)} models")
        
        if not evaluation_results:
            return {"rankings": [], "recommendations": [], "comparison_table": []}
        
        # Extract primary metrics
        model_scores = []
        for result in evaluation_results:
            score = result["metrics"].get("primary_metric", 0.0)
            model_scores.append({
                "model_name": result["model_name"],
                "score": score,
                "metrics": result["metrics"],
                "task_type": result["task_type"]
            })
        
        # Sort by score (higher is better for accuracy/r2, lower is better for MSE)
        task_type = evaluation_results[0]["task_type"]
        if task_type == "classification":
            model_scores.sort(key=lambda x: x["score"], reverse=True)
        else:  # regression - assuming we're using R2 as primary metric
            model_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Create rankings
        rankings = []
        for i, model_data in enumerate(model_scores):
            rankings.append({
                "rank": i + 1,
                "model_name": model_data["model_name"],
                "score": model_data["score"],
                "metrics": model_data["metrics"]
            })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(rankings, task_type)
        
        # Create comparison table
        comparison_table = self._create_comparison_table(rankings)
        
        return {
            "rankings": rankings,
            "recommendations": recommendations,
            "comparison_table": comparison_table,
            "best_model": rankings[0]["model_name"] if rankings else None
        }
    
    def _generate_recommendations(self, rankings: List[Dict], task_type: str) -> List[str]:
        """Generate deployment recommendations."""
        if not rankings:
            return ["No models available for comparison."]
        
        recommendations = []
        best_model = rankings[0]
        
        recommendations.append(f"🏆 **{best_model['model_name']}** is recommended for production deployment.")
        
        if task_type == "classification":
            accuracy = best_model['metrics'].get('accuracy', 0)
            if accuracy > 0.9:
                recommendations.append("✅ Excellent accuracy - suitable for high-stakes decisions.")
            elif accuracy > 0.8:
                recommendations.append("✅ Good accuracy - suitable for most applications.")
            else:
                recommendations.append("⚠️ Moderate accuracy - consider feature engineering or more data.")
        else:
            r2 = best_model['metrics'].get('r2_score', 0)
            if r2 > 0.8:
                recommendations.append("✅ Excellent fit - explains most variance in the data.")
            elif r2 > 0.6:
                recommendations.append("✅ Good fit - suitable for most applications.")
            else:
                recommendations.append("⚠️ Moderate fit - consider additional features or model complexity.")
        
        # Compare top models
        if len(rankings) > 1:
            second_best = rankings[1]
            score_diff = abs(best_model['score'] - second_best['score'])
            if score_diff < 0.02:  # Very close performance
                recommendations.append(f"📊 Consider {second_best['model_name']} as alternative - similar performance.")
        
        return recommendations
    
    def _create_comparison_table(self, rankings: List[Dict]) -> List[Dict]:
        """Create a structured comparison table."""
        table = []
        for ranking in rankings:
            row = {
                "rank": ranking["rank"],
                "model_name": ranking["model_name"],
                "primary_score": ranking["score"]
            }
            # Add all metrics
            row.update(ranking["metrics"])
            table.append(row)
        
        return table