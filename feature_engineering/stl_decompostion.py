import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL

def add_stl_components(df):
    
    
    df = df.copy()
    
    df =df.sort_values(["region","timestamp"]).reset_index(drop=True)
    
    df["trend_stl"] = np.nan
    df["seasonal_stl"] = np.nan
    df["resid_stl"] = np.nan
    
    
    for region, group in df.groupby("region"):
        idx = group.index
        
        
        load = group["load_mw"].astype(float)
        
        load_filled = load.interpolate(method = "linear").ffill().bfill()
        
        stl = STL(load_filled, period = 24, robust = True)
        res =stl.fit()
        
        df.loc[idx, "trend_stl"] = res.trend
        df.loc[idx, "seasonal_stl"] = res.seasonal
        df.loc[idx, "resid_stl"] = res.resid
        
    return df
        