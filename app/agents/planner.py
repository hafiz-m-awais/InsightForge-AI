from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import AgentState
from app.agents.llm_router import get_llm
import json

def planner_node(state: AgentState) -> dict:
    """
    Parses user intent and creates an execution plan.
    """
    print("--- PLANNER NODE ---")
    intent = state.get("user_intent", "Perform full analysis")
    
    # We use a structured prompt to ask the LLM for a plan
    llm = get_llm()
    
    system_prompt = (
        "You are the Planner Agent for an autonomous data science system. "
        "Return only a JSON array of strings representing pipeline steps. "
        "Typical steps: [\"EDA\", \"Feature Engineering\", \"Model Training\", \"Evaluation\", \"Insights\"]"
    )
    human_message = f"The user's intent is: \"{intent}\". Create a step-by-step pipeline plan."

    try:
        # Simplistic parsing for MVP
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ])
        # Strip markdown if present
        clean_response = str(response.content).strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:-3]
        plan = json.loads(clean_response)
    except Exception as e:
        print(f"Planner fallback. Error: {e}")
        plan = ["EDA", "Feature Engineering", "Model Training", "Evaluation", "Insights"]

    return {"pipeline_plan": plan}
