from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
import json

# Import nodes
from app.agents.planner import planner_node
from app.agents.eda import eda_node
from app.agents.feature_engineering import feature_engineering_node
from app.agents.ml_agent import ml_agent_node
from app.agents.evaluator import evaluator_node
from app.agents.critic import critic_node
from app.agents.insight import insight_node
from app.agents.report import report_node

def should_iterate(state: AgentState):
    feedback = state.get("critic_feedback", '{"action": "continue"}')
    try:
        feedback_data = json.loads(feedback)
        if feedback_data.get("action") == "retry":
            return "feature_engineering"
    except json.JSONDecodeError:
        if "REJECT" in str(feedback).upper() or "RETRY" in str(feedback).upper():
            return "feature_engineering"
    return "insight"

def build_graph():
    """
    Constructs the LangGraph state machine.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("ml_agent", ml_agent_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("insight", insight_node)
    workflow.add_node("report", report_node)
    
    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "eda")
    workflow.add_edge("eda", "feature_engineering")
    workflow.add_edge("feature_engineering", "ml_agent")
    workflow.add_edge("ml_agent", "evaluator")
    workflow.add_edge("evaluator", "critic")
    
    # Conditional edge after critic
    workflow.add_conditional_edges(
        "critic",
        should_iterate,
        {
            "feature_engineering": "feature_engineering",
            "insight": "insight"
        }
    )
    
    workflow.add_edge("insight", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()

# Global graph instance
ds_agent_graph = build_graph()
