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
        
        # Identify categorical and numerical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Simple processing
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = df[col].fillna('Missing')
            df[col] = le.fit_transform(df[col].astype(str))
            
        scaler = StandardScaler()
        for col in num_cols:
            # fill numerical NA with median
            df[col] = df[col].fillna(df[col].median())
            # We skip scaling target if it's the last column, but for MVP we might just scale all features except the last
            
        # Assume last column is target for simplicity
        target_col = df.columns[-1]
        features = list(df.columns[:-1])
        
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
