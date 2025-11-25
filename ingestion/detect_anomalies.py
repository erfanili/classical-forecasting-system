import pandas as pd 
import numpy as np 

def detect_spikes(df, z_thresh = 4.):
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['load_mw'] = df['load_mw'].astype(float)
    df['is_anomaly'] = False
    
    for region, group in df.groupby('region'):
        idx = group.index
        
        diff = group['load_mw'].diff()
        mean = diff.mean()
        std = diff.std()
        
        if std == 0 or np.isnan(std):
            z = pd.Series(0, index=diff.index)
        else:
            z = (diff - mean) / std
            
        anomalies = z.abs() > z_thresh
        
        df.loc[idx,'is_anomaly']= anomalies
        
    return df
        
        
