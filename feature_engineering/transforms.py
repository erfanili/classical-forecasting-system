import pandas as pd
import numpy as np
from scipy.stats import boxcox


def add_boxcox_transform(df, per_region):
    
    df = df.copy()
    
    
    df["load_mw"] = df["load_mw"].astype(float)
    eps = 1e-6
    
    df["load_pos"] = df["load_mw"].clip(lower=eps)
    df["load_bc"] = np.nan
    df["lambda_bc"] = np.nan
    
    if per_region:
        
        for region, group in df.groupby("region"):
            
            idx = group.index
            
            load_values = group["load_pos"].values
            bc_values, lam = boxcox(load_values)
            
            df.loc[idx, "load_bc"] = bc_values
            df.loc[idx, "lambda_bc"] = lam
            
            
    else:
        
        load_values = df["load_pos"].values
        bc_values, lam = boxcox(load_values)
        
        df["load_bc"] = bc_values
        df["lambda_bc"] = lam
    
    df = df.drop(columns = ["load_pos"])
    
    return df