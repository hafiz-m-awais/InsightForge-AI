from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import AgentState
from app.agents.llm_router import get_llm
import json

def planner_node(state: AgentState) -> dict:
    """
    Parses user intent and creates an execution plan.
    """
    print(f"--- PLANNER NODE ---")
    intent = state.get("user_intent", "Perform full analysis")
    
    # We use a structured prompt to ask the LLM for a plan
    llm = get_llm()
    prompt = f"""
    You are the Planner Agent for an autonomous data science system.
    The user's intent is: "{intent}"
    Create a high-level step-by-step pipeline plan to achieve this intent.
    Return a JSON array of strings representing the steps. 
    Typical steps: ["EDA", "Feature Engineering", "Model Training", "Evaluation", "Insights"]
    """
    
    try:
        # Simplistic parsing for MVP
        response = llm.invoke([SystemMessage(content=prompt)])
        # Strip markdown if present
        clean_response = response.content.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:-3]
        plan = json.loads(clean_response)
    except Exception as e:
        print(f"Planner fallback. Error: {e}")
        plan = ["EDA", "Feature Engineering", "Model Training", "Evaluation", "Insights"]

    return {"pipeline_plan": plan}
