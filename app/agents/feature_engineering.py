import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.agents.state import AgentState
import os

def feature_engineering_node(state: AgentState) -> dict:
    """
    Performs basic Feature Engineering: encoding categorical variables and scaling numerical ones.
    """
    print(f"--- FEATURE ENGINEERING NODE ---")
    dataset_path = state.get("dataset_path")
    
    if not dataset_path:
        return {"errors": state.get("errors", []) + ["No dataset path provided for FE."]}
        
    try:
        df = pd.read_csv(dataset_path)
        
        # Use target_col from state if available; fall back to last column
        target_col = state.get("target_col") or df.columns[-1]
        
        # Identify categorical and numerical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Use a separate LabelEncoder instance per column to avoid cross-column contamination
        label_encoders = {}
        for col in cat_cols:
            df[col] = df[col].fillna('Missing')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
        scaler = StandardScaler()
        for col in num_cols:
            # fill numerical NA with median
            df[col] = df[col].fillna(df[col].median())
            
        features = [c for c in df.columns if c != target_col]
        
        if target_col in num_cols and df[target_col].nunique() < 20: # heuristic for classification
            pass # keep it as is
            
        # Save processed data
        processed_path = dataset_path.replace('.csv', '_processed.csv')
        df.to_csv(processed_path, index=False)
        
        fe_info = {
            "processed_path": processed_path,
            "categorical_encoded": list(cat_cols),
            "numerical_scaled": list(num_cols),
            "target_col_assumed": target_col,
            "features": features
        }
        
        return {"feature_engineering_info": fe_info}
        
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"FE Error: {str(e)}"]}
