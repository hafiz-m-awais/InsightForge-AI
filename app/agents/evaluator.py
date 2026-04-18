from app.agents.state import AgentState
import joblib
import os

def evaluator_node(state: AgentState) -> dict:
    """
    Evaluates trained models and selects the best one.
    """
    print(f"--- EVALUATOR NODE ---")
    results = state.get("model_results", [])
    
    if not results:
        return {"errors": state.get("errors", []) + ["No model results provided to evaluator."]}
        
    try:
        # Find best model
        best_model_info = max(results, key=lambda x: x["score"])
        
        best_model = best_model_info["model_instance"]
        best_name = best_model_info["model_name"]
        best_score = best_model_info["score"]
        
        print(f"Best model selected: {best_name} with score {best_score}")
        
        # We can just keep the best model info in state and save it
        model_path = f"models/best_model_{best_name}.joblib"
        joblib.dump(best_model, model_path)
        
        return {
            "best_model_path": model_path,
            "insights": f"The best performing model was {best_name} achieving a score of {best_score:.4f}."
        }
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Evaluator Error: {str(e)}"]}
