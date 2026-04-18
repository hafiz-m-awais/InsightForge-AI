"""
Model evaluation and report generation routes.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import asyncio
import logging
import joblib
import tempfile
import pandas as pd
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from app.agents.ml_training_agent import MLTrainingAgent

router = APIRouter(prefix="/api", tags=["evaluation"])
logger = logging.getLogger(__name__)

# Bounded thread-pool shared across all evaluation handlers
_executor = ThreadPoolExecutor(max_workers=4)

UPLOAD_DIR = Path("datasets").resolve()


def _safe_dataset_path(raw: str) -> Path:
    """Resolve a client-supplied path safely inside UPLOAD_DIR."""
    candidate = (UPLOAD_DIR / Path(raw).name).resolve()
    if not candidate.is_relative_to(UPLOAD_DIR):
        raise ValueError("Invalid dataset path")
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluationRequest(BaseModel):
    dataset_path: str
    target_col: str
    tuning_results: dict
    metrics: list[str]
    test_size: float = 0.2
    include_visualizations: bool = True
    include_feature_importance: bool = True


@router.post("/model-evaluation")
async def model_evaluation(request: ModelEvaluationRequest):
    """Evaluate tuned models with comprehensive metrics."""
    try:
        ml_agent = MLTrainingAgent()

        try:
            dataset_path = _safe_dataset_path(request.dataset_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid dataset path")
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(_executor, pd.read_csv, dataset_path)

        X_train, X_val, X_test, y_train, y_val, y_test = ml_agent.prepare_data(
            df, request.target_col, test_size=request.test_size, validation_size=0.15
        )

        tuning_results = request.tuning_results
        if not tuning_results or "results" not in tuning_results:
            raise HTTPException(status_code=400, detail="Invalid tuning results")

        evaluation_results = []
        for model_result in tuning_results["results"]:
            model_name = model_result.get("model_name")
            best_params = model_result.get("best_params", {})

            if not model_name:
                continue

            model_name = ml_agent._normalize_model_name(model_name)

            try:
                model_dict = (
                    ml_agent.classification_models
                    if ml_agent.task_type == "classification"
                    else ml_agent.regression_models
                )

                if model_name not in model_dict:
                    logger.warning(f"Model {model_name} not found, skipping")
                    continue

                import inspect
                base_model = model_dict[model_name]
                model_sig = inspect.signature(base_model.__class__.__init__)
                valid_params = set(model_sig.parameters.keys()) - {"self"}
                filtered_params = {k: v for k, v in best_params.items() if k in valid_params}
                if "random_state" in valid_params and "random_state" not in filtered_params:
                    filtered_params["random_state"] = ml_agent.random_state
                model = base_model.__class__(**filtered_params)

                model.fit(X_train, y_train)

                # Use a temp file to avoid unbounded disk growth in models/
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".joblib")
                os.close(tmp_fd)
                try:
                    joblib.dump(model, tmp_path)
                    eval_result = await loop.run_in_executor(
                        _executor, ml_agent.evaluate_model, tmp_path,
                        X_test, y_test, model_name, X_train,
                    )
                finally:
                    os.unlink(tmp_path)
                evaluation_results.append(eval_result)

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue

        if not evaluation_results:
            raise HTTPException(status_code=400, detail="No models could be evaluated")

        return {
            "job_id": f"eval_{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "evaluation_results": evaluation_results,
            "dataset_info": {
                "total_samples": len(df),
                "test_samples": len(X_test),
                "features": len(X_test.columns),
            },
        }

    except Exception as e:
        logger.error(f"Model evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Report (mock — legacy)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/evaluation-report")
async def evaluation_report(request: dict = {}):
    """Legacy mock endpoint — replaced by /api/evaluation-report-real."""
    return JSONResponse(
        status_code=410,
        content={
            "deprecated": True,
            "warning": "This endpoint returns random/fake data and has been retired.",
            "redirect": "/api/evaluation-report-real",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real Evaluation Report
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/evaluation-report-real")
async def evaluation_report_real(request: dict = {}):
    """Generate comprehensive HTML evaluation report from actual evaluation results."""
    try:
        evaluation_results = request.get("evaluation_results", [])
        comparison_results = request.get("comparison_results", {})
        dataset_name = request.get("dataset_name", "Dataset")

        if not evaluation_results:
            raise HTTPException(status_code=400, detail="No evaluation results provided")

        html_content = generate_evaluation_report_html_real(
            evaluation_results, comparison_results, dataset_name
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/evaluation_report_{timestamp}.html"

        os.makedirs("reports", exist_ok=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, lambda: open(report_filename, "w", encoding="utf-8").write(html_content))

        return {
            "status": "completed",
            "report_path": report_filename,
            "report_html": html_content,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_evaluation_report_html_real(
    evaluation_results: list, comparison_results: dict, dataset_name: str
) -> str:
    """Generate HTML report from actual evaluation results."""
    css = """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }
        .header h1 { margin: 0; font-size: 2.5rem; font-weight: 300; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1rem; }
        .section { padding: 30px; border-bottom: 1px solid #e1e5e9; }
        .section:last-child { border-bottom: none; }
        .section h2 { color: #2c3e50; margin-bottom: 20px; font-size: 1.8rem; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-left: 4px solid #667eea; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px; }
        .model-row { display: flex; justify-content: space-between; align-items: center; padding: 15px; margin: 10px 0; background: #f8f9fa; border-radius: 5px; }
        .model-name { font-weight: bold; color: #2c3e50; }
        .model-score { font-size: 1.2rem; color: #27ae60; }
        .rank-badge { background: #667eea; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .feature-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #ecf0f1; }
        .importance-bar { width: 200px; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }
        .importance-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); }
        .recommendations { background: #e8f5e8; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .recommendations h3 { color: #27ae60; margin-bottom: 15px; }
        .timestamp { text-align: center; padding: 20px; color: #7f8c8d; font-style: italic; }
    """

    html_parts = [
        "<!DOCTYPE html>", "<html lang='en'>", "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Model Evaluation Report</title>",
        f"<style>{css}</style>",
        "</head>", "<body>",
        "<div class='container'>",
        "<div class='header'>",
        "<h1>Model Evaluation Report</h1>",
        f"<p>Dataset: {dataset_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "</div>",
    ]

    best_model_raw = comparison_results.get("best_model", "Unknown")
    if isinstance(best_model_raw, dict):
        best_model = best_model_raw.get("model_name", "Unknown")
    else:
        best_model = best_model_raw

    if best_model != "Unknown" and evaluation_results:
        best_result = next(
            (r for r in evaluation_results if r.get("model_name", "") == best_model),
            evaluation_results[0],
        )
        html_parts += [
            "<div class='section'>",
            "<h2>🏆 Best Performing Model</h2>",
            f"<p style='font-size:1.3rem;margin-bottom:20px;'><strong>{best_model}</strong> achieved the highest performance.</p>",
            "<div class='metrics-grid'>",
        ]
        for metric, value in best_result["metrics"].items():
            if metric != "primary_metric":
                html_parts += [
                    "<div class='metric-card'>",
                    f"<div class='metric-value'>{value:.4f}</div>",
                    f"<div class='metric-label'>{metric.replace('_', ' ').title()}</div>",
                    "</div>",
                ]
        html_parts += ["</div>", "</div>"]

    rankings = comparison_results.get("rankings", [])
    if rankings:
        html_parts += ["<div class='section'>", "<h2>📊 Model Comparison</h2>"]
        for rank_data in rankings:
            html_parts += [
                "<div class='model-row'>",
                f"<div class='rank-badge'>{rank_data['rank']}</div>",
                f"<div class='model-name'>{rank_data['model_name']}</div>",
                f"<div class='model-score'>{rank_data['score']:.4f}</div>",
                "</div>",
            ]
        html_parts.append("</div>")

    if evaluation_results and evaluation_results[0].get("feature_importance"):
        feature_importance = evaluation_results[0]["feature_importance"][:10]
        max_importance = feature_importance[0]["importance"] if feature_importance else 1
        html_parts += [
            "<div class='section'>",
            f"<h2>🔍 Feature Importance - {best_model}</h2>",
            "<div>",
        ]
        for feature in feature_importance:
            width_pct = (feature["importance"] / max_importance) * 100
            html_parts += [
                "<div class='feature-item'>",
                f"<span>{feature['feature']}</span>",
                "<div style='display:flex;align-items:center;'>",
                f"<span style='margin-right:10px;font-weight:bold;'>{feature['importance']:.4f}</span>",
                "<div class='importance-bar'>",
                f"<div class='importance-fill' style='width:{width_pct:.1f}%;'></div>",
                "</div></div></div>",
            ]
        html_parts += ["</div>", "</div>"]

    recommendations = comparison_results.get("recommendations", [])
    if recommendations:
        html_parts += [
            "<div class='section'>", "<h2>💡 Recommendations</h2>",
            "<div class='recommendations'>",
        ]
        for rec in recommendations:
            html_parts.append(f"<p>{rec}</p>")
        html_parts += [
            "<h3>Next Steps</h3>", "<ul>",
            "<li>Conduct additional testing on out-of-sample data</li>",
            "<li>Monitor model performance in production environment</li>",
            "<li>Set up automated retraining pipeline</li>",
            "<li>Implement model explainability dashboard</li>",
            "</ul>", "</div>", "</div>",
        ]

    html_parts += [
        "<div class='timestamp'>",
        f"Report generated by InsightForge-AI on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        "</div>",
        "</div>", "</body>", "</html>",
    ]

    return "\n".join(html_parts)
