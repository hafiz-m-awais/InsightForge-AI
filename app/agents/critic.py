from app.agents.state import AgentState
from langchain_core.messages import SystemMessage
from app.agents.llm_router import get_llm
from pydantic import BaseModel, Field
import json

class CriticDecision(BaseModel):
    action: str = Field(description="Must be 'retry' if the model underfits/overfits and needs improvement, or 'continue' if it is satisfactory.")
    reason: str = Field(description="Brief explanation of the decision.")

def critic_node(state: AgentState) -> dict:
    """
    Reviews the evaluation and decides if iteration is needed.
    """
    print(f"--- CRITIC NODE ---")
    insights = state.get("insights", "")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 1)
    
    # In a full system, the LLM criticizes the model performance and dataset features.
    llm = get_llm()
    
    try:
        structured_llm = llm.with_structured_output(CriticDecision)
        
        prompt = f"""
        Review the following result from a machine learning pipeline:
        {insights}
        
        Does this result look satisfactory or is there potential for overfitting/underfitting?
        Evaluate and decide whether to 'retry' or 'continue'.
        """
        
        response = structured_llm.invoke([SystemMessage(content=prompt)])
        action = response.action.lower()
        reason = response.reason
        
        if action == "retry" and iteration_count < max_iterations:
            return {
                "critic_feedback": json.dumps({"action": "retry", "reason": reason}),
                "iteration_count": iteration_count + 1
            }
            
        return {"critic_feedback": json.dumps({"action": "continue", "reason": reason})}
    except Exception as e:
        # Fallback handling
        try:
            prompt_fallback = f"Review this: {insights}\nRespond strictly with JSON: {{\"action\": \"continue\" or \"retry\", \"reason\": \"...\"}}"
            res = llm.invoke([SystemMessage(content=prompt_fallback)]).content.strip()
            if "retry" in res.lower() and iteration_count < max_iterations:
                return {
                    "critic_feedback": json.dumps({"action": "retry", "reason": "Parsed from string fallback"}),
                    "iteration_count": iteration_count + 1
                }
        except Exception:
            pass
        return {"critic_feedback": json.dumps({"action": "continue", "reason": "Error parsing LLM feedback"})}
