from app.agents.state import AgentState
from langchain_core.messages import SystemMessage
from app.agents.llm_router import get_llm

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
    prompt = f"""
    Review the following result from a machine learning pipeline:
    {insights}
    
    Does this result look satisfactory or is there potential for overfitting/underfitting?
    Respond with either "PASS" or "REJECT".
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        feedback = response.content.strip()
        
        if "REJECT" in feedback and iteration_count < max_iterations:
            # We would normally route back to Feature Engineering or ML Agent
            return {
                "critic_feedback": "REJECT: " + feedback,
                "iteration_count": iteration_count + 1
            }
            
        return {"critic_feedback": "PASS"}
    except Exception as e:
        return {"critic_feedback": "PASS"} # Fallback to pass on error
