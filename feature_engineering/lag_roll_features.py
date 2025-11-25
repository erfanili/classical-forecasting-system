import pandas as pd




def add_lag_features(df, lags):
    df = df.copy()
    
    
    
    df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
    
    for k in lags:
        lag_col = f"lage_{k}"
        df[lag_col] = (df.groupby("region")["load_mw"].shift(k)
        )
        
        
        return df
    
    
def add_rolling_features(df, windows):
    
    df = df.copy()
    
    df.sort_values(["region", "timestamp"]).reset_index(drop=True)
    
    for w in windows:
        
        grouped = df.groupby(["region"])["load_mw"]
        
        df[f"load_mean_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods = 1).mean())
        df[f"load_std_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods = 1).std())
        
    return df