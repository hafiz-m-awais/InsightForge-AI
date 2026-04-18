from app.agents.state import AgentState
from langchain_core.messages import SystemMessage
from app.agents.llm_router import get_llm

def insight_node(state: AgentState) -> dict:
    """
    Generates business insights from technical output.
    """
    print(f"--- INSIGHT NODE ---")
    eda = state.get("eda_summary", {})
    insights_text = state.get("insights", "")
    intent = state.get("user_intent", "")
    
    llm = get_llm()
    prompt = f"""
    Given the original user intent: "{intent}",
    and the EDA summary: {eda.get('numerical_summary')},
    and the model result: {insights_text}.
    
    Write a 2-3 paragraph business-focused summary of these findings, explaining key drivers and recommendations.
    Format as clean markdown.
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        final_insight = response.content.strip()
        return {"insights": insights_text + "\n\n### Business Insights:\n" + final_insight}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Insight Error: {str(e)}"]}
