"""
Advanced analysis routes: feature selection and leakage detection.
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import asyncio
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    chi2, f_classif, f_regression, RFE, SelectFromModel,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["analysis"])

UPLOAD_DIR = "datasets"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSelectionRequest(BaseModel):
    dataset_path: str
    target_col: str
    method: str = "correlation"  # correlation | mutual_info | chi2 | anova_f | rfe | lasso | tree_importance
    n_features: int = 10
    correlation_threshold: float = 0.1
    variance_threshold: float = 0.01


@router.post("/feature-selection")
async def feature_selection(request: FeatureSelectionRequest):
    """Run feature selection using various methods."""
    try:
        dataset_path = Path(UPLOAD_DIR) / request.dataset_path
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found on server.")

        df = pd.read_csv(dataset_path)

        X = df.drop(columns=[request.target_col])
        y = df[request.target_col]

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        label_encoders: dict = {}
        X_processed = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = np.array(le.fit_transform(X[col].astype(str)), dtype=int)
            label_encoders[col] = le

        is_classification = y.dtype == "object" or len(y.unique()) < 20
        y_encoded = np.array(LabelEncoder().fit_transform(y)) if (is_classification and y.dtype == "object") else y

        selected_features: list = []
        importance_scores: dict = {}
        method_info: dict = {}

        if request.variance_threshold > 0:
            variance_selector = VarianceThreshold(threshold=request.variance_threshold)
            variance_selector.fit_transform(X_processed)
            variance_features = X_processed.columns[variance_selector.get_support()].tolist()
            X_processed = X_processed[variance_features]
            method_info["variance_threshold"] = {
                "removed_count": len(X.columns) - len(variance_features),
                "remaining_features": variance_features,
            }

        await asyncio.sleep(1)

        if request.method == "correlation":
            corr_matrix = X_processed.corrwith(pd.Series(y_encoded)) if is_classification else X_processed.corrwith(y)
            corr_scores = abs(corr_matrix).sort_values(ascending=False)
            selected_features = corr_scores[corr_scores >= request.correlation_threshold].head(request.n_features).index.tolist()
            importance_scores = corr_scores.to_dict()
            method_info["correlation"] = {
                "threshold": request.correlation_threshold,
                "max_correlation": float(corr_scores.max()),
                "min_correlation": float(corr_scores.min()),
            }

        elif request.method == "mutual_info":
            scores = (
                mutual_info_classif(X_processed, y_encoded, random_state=42)
                if is_classification
                else mutual_info_regression(X_processed, y_encoded, random_state=42)
            )
            feature_scores = pd.Series(scores, index=X_processed.columns).sort_values(ascending=False)
            selected_features = feature_scores.head(request.n_features).index.tolist()
            importance_scores = feature_scores.to_dict()
            method_info["mutual_info"] = {
                "max_score": float(feature_scores.max()),
                "mean_score": float(feature_scores.mean()),
            }

        elif request.method == "chi2" and is_classification:
            X_positive = X_processed.copy()
            for col in X_positive.columns:
                if X_positive[col].min() < 0:
                    X_positive[col] = X_positive[col] - X_positive[col].min()
            selector = SelectKBest(chi2, k=min(request.n_features, len(X_positive.columns)))
            selector.fit(X_positive, y_encoded)
            selected_features = X_positive.columns[selector.get_support()].tolist()
            importance_scores = {col: float(score) for col, score in zip(X_positive.columns, list(selector.scores_))}  # type: ignore
            method_info["chi2"] = {
                "selected_count": len(selected_features),
                "max_score": float(max(list(selector.scores_))),  # type: ignore
            }

        elif request.method == "anova_f":
            selector = SelectKBest(
                f_classif if is_classification else f_regression,
                k=min(request.n_features, len(X_processed.columns)),
            )
            selector.fit(X_processed, y_encoded)
            selected_features = X_processed.columns[selector.get_support()].tolist()
            importance_scores = {col: float(score) for col, score in zip(X_processed.columns, list(selector.scores_))}  # type: ignore
            method_info["anova_f"] = {
                "selected_count": len(selected_features),
                "max_f_score": float(max(list(selector.scores_))),  # type: ignore
            }

        elif request.method == "rfe":
            estimator = LogisticRegression(random_state=42, max_iter=1000) if is_classification else Lasso(random_state=42)
            rfe = RFE(estimator, n_features_to_select=min(request.n_features, len(X_processed.columns)))
            rfe.fit(X_processed, y_encoded)
            selected_features = X_processed.columns[rfe.support_].tolist()
            importance_scores = dict(zip(X_processed.columns, rfe.ranking_))
            method_info["rfe"] = {
                "selected_count": len(selected_features),
                "estimator": type(estimator).__name__,
            }

        elif request.method == "lasso":
            lasso = Lasso(alpha=0.01, random_state=42)
            selector = SelectFromModel(lasso)
            selector.fit(X_processed, y_encoded)
            selected_features = X_processed.columns[selector.get_support()].tolist()
            lasso.fit(X_processed, y_encoded)
            importance_scores = dict(zip(X_processed.columns, abs(lasso.coef_)))
            method_info["lasso"] = {
                "alpha": 0.01,
                "selected_count": len(selected_features),
                "non_zero_coef": int(np.sum(lasso.coef_ != 0)),
            }

        elif request.method == "tree_importance":
            model = (
                RandomForestClassifier(n_estimators=100, random_state=42)
                if is_classification
                else RandomForestRegressor(n_estimators=100, random_state=42)
            )
            model.fit(X_processed, y_encoded)
            feature_importance = pd.Series(model.feature_importances_, index=X_processed.columns).sort_values(ascending=False)
            selected_features = feature_importance.head(request.n_features).index.tolist()
            importance_scores = feature_importance.to_dict()
            method_info["tree_importance"] = {
                "model_type": type(model).__name__,
                "n_estimators": 100,
                "max_importance": float(feature_importance.max()),
            }

        # Fallback
        if not selected_features:
            corr_matrix = X_processed.corrwith(pd.Series(y_encoded))
            corr_scores = abs(corr_matrix).sort_values(ascending=False)
            selected_features = corr_scores.head(min(request.n_features, len(X_processed.columns))).index.tolist()
            importance_scores = corr_scores.to_dict()
            method_info["fallback"] = "Used correlation due to method failure"

        original_feature_count = len(X.columns)
        selected_feature_count = len(selected_features)
        reduction_percentage = ((original_feature_count - selected_feature_count) / original_feature_count) * 100
        sorted_importance = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        await asyncio.sleep(1)

        return {
            "status": "success",
            "method": request.method,
            "original_feature_count": original_feature_count,
            "selected_feature_count": selected_feature_count,
            "reduction_percentage": round(reduction_percentage, 2),
            "selected_features": selected_features,
            "importance_scores": {k: float(v) for k, v in importance_scores.items()},
            "importance_ranking": [(k, float(v)) for k, v in sorted_importance],
            "method_info": method_info,
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
            "is_classification": is_classification,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Detection
# ─────────────────────────────────────────────────────────────────────────────

class LeakageDetectionRequest(BaseModel):
    dataset_path: str
    target_col: str
    correlation_threshold: float = 0.8
    perfect_correlation_threshold: float = 0.95


@router.post("/leakage-detection")
async def leakage_detection(request: LeakageDetectionRequest):
    """Detect potential data leakage in features."""
    try:
        dataset_path = Path(UPLOAD_DIR) / request.dataset_path
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found on server.")

        df = pd.read_csv(dataset_path)

        X = df.drop(columns=[request.target_col])
        y = df[request.target_col]

        X_processed = X.copy()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = np.array(le.fit_transform(X[col].astype(str)), dtype=int)

        y_encoded = np.array(LabelEncoder().fit_transform(y)) if y.dtype == "object" else y

        await asyncio.sleep(2)

        correlations = X_processed.corrwith(pd.Series(y_encoded))

        detections = []
        high_risk_features: list = []
        medium_risk_features: list = []
        low_risk_features: list = []

        for feature, correlation in correlations.items():
            abs_corr = abs(correlation)

            if abs_corr >= request.perfect_correlation_threshold:
                risk_level = "high"
                issue_type = "perfect_correlation"
                description = f"Perfect or near-perfect correlation ({abs_corr:.3f}) with target variable"
                recommendation = "Remove this feature — likely contains target information"
                high_risk_features.append(feature)
            elif abs_corr >= request.correlation_threshold:
                risk_level = "medium"
                issue_type = "high_correlation"
                description = f"High correlation ({abs_corr:.3f}) with target variable"
                recommendation = "Investigate if this feature is calculated using target information"
                medium_risk_features.append(feature)
            elif abs_corr >= 0.5:
                risk_level = "low"
                issue_type = "moderate_correlation"
                description = f"Moderate correlation ({abs_corr:.3f}) with target variable"
                recommendation = "Monitor this feature during validation"
                low_risk_features.append(feature)
            else:
                continue

            if abs_corr >= 0.5:
                detections.append({
                    "feature": feature,
                    "target_correlation": float(correlation),
                    "risk_level": risk_level,
                    "issue_type": issue_type,
                    "description": description,
                    "recommendation": recommendation,
                })

        temporal_keywords = ["next", "future", "after", "following", "subsequent", "later", "post"]
        for feature in X.columns:
            if any(kw in feature.lower() for kw in temporal_keywords):
                detections.append({
                    "feature": feature,
                    "target_correlation": float(correlations.get(feature, 0)),
                    "risk_level": "high",
                    "issue_type": "temporal_leakage",
                    "description": "Feature name suggests future information not available at prediction time",
                    "recommendation": "Remove this feature — contains data from after the target event",
                })
                if feature not in high_risk_features:
                    high_risk_features.append(feature)

        target_variations = ["_derived", "_calculated", "_flag", "_indicator", "_score"]
        target_base = request.target_col.lower().replace("_", "")
        for feature in X.columns:
            feature_lower = feature.lower()
            if target_base in feature_lower or any(v in feature_lower for v in target_variations):
                abs_corr = abs(correlations.get(feature, 0))
                if abs_corr > 0.7:
                    detections.append({
                        "feature": feature,
                        "target_correlation": float(correlations.get(feature, 0)),
                        "risk_level": "high",
                        "issue_type": "direct_leakage",
                        "description": "Appears to be directly derived from target variable",
                        "recommendation": "Remove immediately — this is the target in disguise",
                    })
                    if feature not in high_risk_features:
                        high_risk_features.append(feature)

        high_risk_features = list(set(high_risk_features))
        medium_risk_features = list(set(medium_risk_features))
        low_risk_features = list(set(low_risk_features))
        all_flagged = set(high_risk_features + medium_risk_features + low_risk_features)
        clean_features = [f for f in X.columns if f not in all_flagged]

        overall_risk = "high" if high_risk_features else ("medium" if medium_risk_features else "clean")

        recommendations = []
        if high_risk_features:
            recommendations.append(f"Remove {len(high_risk_features)} high-risk features before training")
        if medium_risk_features:
            recommendations.append(f"Investigate {len(medium_risk_features)} medium-risk features for potential issues")
        recommendations.extend([
            "Validate data collection timeline to prevent temporal leakage",
            "Review feature engineering process for target leakage",
            "Use proper train/validation split with temporal considerations if applicable",
        ])

        await asyncio.sleep(1)

        return {
            "status": "success",
            "total_features_checked": len(X.columns),
            "high_risk_features": high_risk_features,
            "medium_risk_features": medium_risk_features,
            "low_risk_features": low_risk_features,
            "clean_features": clean_features,
            "detections": detections,
            "overall_risk": overall_risk,
            "recommendations": recommendations,
            "correlation_threshold": request.correlation_threshold,
            "perfect_correlation_threshold": request.perfect_correlation_threshold,
            "analysis_summary": {
                "high_risk_count": len(high_risk_features),
                "medium_risk_count": len(medium_risk_features),
                "low_risk_count": len(low_risk_features),
                "clean_count": len(clean_features),
                "total_detections": len(detections),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
