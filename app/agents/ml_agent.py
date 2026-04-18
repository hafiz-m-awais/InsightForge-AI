import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from app.agents.state import AgentState

def ml_agent_node(state: AgentState) -> dict:
    """
    Trains multiple models based on the processed dataset.
    """
    print(f"--- ML AGENT NODE ---")
    fe_info = state.get("feature_engineering_info", {})
    processed_path = fe_info.get("processed_path")
    target_col = fe_info.get("target_col_assumed")
    
    if not processed_path or not target_col:
        return {"errors": state.get("errors", []) + ["Missing feature engineering info for ML Agent."]}
        
    try:
        df = pd.read_csv(processed_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Simple heuristic for classification vs regression
        is_classification = df[target_col].nunique() < 20
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {}
        if is_classification:
            models = {
                "RandomForest": RandomForestClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42)
            }
        else:
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42)
            }
            
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test) # Accuracy for classification, R2 for regression
            results.append({
                "model_name": name,
                "score": score,
                "model_instance": model, # In a real distributed state we might not pass the object directly, but for local it's fine
                "is_classification": is_classification,
                "X_test": X_test,
                "y_test": y_test
            })
            
        return {"model_results": results}
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"ML Agent Error: {str(e)}"]}
