import pandas as pd
from app.agents.state import AgentState

def eda_node(state: AgentState) -> dict:
    """
    Performs Exploratory Data Analysis.
    """
    print(f"--- EDA NODE ---")
    dataset_path = state.get("dataset_path")
    
    if not dataset_path:
        return {"errors": state.get("errors", []) + ["No dataset path provided for EDA."]}
        
    try:
        df = pd.read_csv(dataset_path)
        
        # Basic EDA metrics
        eda_summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numerical_summary": df.describe().to_dict()
        }
        
        return {"eda_summary": eda_summary}
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"EDA Error: {str(e)}"]}
