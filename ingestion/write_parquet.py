import pandas as pd
from pathlib import Path


def save_cleaned_data(df,path):
    
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    
    try:
        df.to_parquet(output_path,index =False)
    except Exception as e:
        raise RuntimeError(f"Failed to write parquet file to {path}: {e}")