from app.agents.state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.llm_router import get_llm

def insight_node(state: AgentState) -> dict:
    """
    Generates business insights from technical output.
    """
    print("--- INSIGHT NODE ---")
    eda = state.get("eda_summary", {})
    insights_text = state.get("insights", "")
    intent = state.get("user_intent", "")
    
    llm = get_llm()
    system_prompt = (
        "You are a senior data scientist writing business-focused summaries of ML pipeline results. "
        "Format your response as clean markdown."
    )
    human_message = (
        f"User intent: \"{intent}\"\n\n"
        f"EDA numerical summary: {eda.get('numerical_summary')}\n\n"
        f"Model result: {insights_text}\n\n"
        "Write a 2-3 paragraph business-focused summary explaining key drivers and recommendations."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message),
        ])
        final_insight = str(response.content).strip()
        return {"insights": insights_text + "\n\n### Business Insights:\n" + final_insight}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Insight Error: {str(e)}"]}
