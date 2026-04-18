"""
Legacy full-pipeline route.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.graph import ds_agent_graph
from app.agents.llm_router import llm_manager, LLMProvider

router = APIRouter(prefix="/api", tags=["pipeline"])


class PipelineRequest(BaseModel):
    dataset_path: str
    user_intent: str
    provider: str = "openrouter"


@router.post("/run-pipeline")
async def run_pipeline(request: PipelineRequest):
    try:
        provider_enum = LLMProvider(request.provider.lower())
        llm_manager.set_default_provider(provider_enum)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid LLM provider.")

    initial_state = {
        "messages": [],
        "user_intent": request.user_intent,
        "dataset_path": request.dataset_path,
        "pipeline_plan": [],
        "current_step": 0,
        "eda_summary": {},
        "feature_engineering_info": {},
        "model_results": [],
        "best_model_path": "",
        "critic_feedback": "",
        "iteration_count": 0,
        "max_iterations": 1,
        "insights": "",
        "report_path": "",
        "errors": [],
    }

    try:
        final_state = ds_agent_graph.invoke(initial_state)  # type: ignore

        if "model_results" in final_state:
            for res in final_state["model_results"]:
                res.pop("model_instance", None)
                res.pop("X_test", None)
                res.pop("y_test", None)

        return {
            "status": "success",
            "report_path": final_state.get("report_path"),
            "best_model_path": final_state.get("best_model_path"),
            "insights": final_state.get("insights"),
            "errors": final_state.get("errors"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
