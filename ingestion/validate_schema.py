import pandas as pd 


import pandas as pd 


def clean(df):
    
    df = df.drop_duplicates(subset=['timestamp','region'])
    
    
    df ['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
    
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    
    full_index = pd.date_range(start=start,end = end, freq = 'H')
    
    cleaned_parts = []
    
    for region, group in df.groupby('region'):
        
        group = group.sort_values('timestamp').set_index("timestamp")
        
        group = group.reindex(full_index)
        
        group.index.name = "timestamp"
        group["region"] = region
        
        group['load_mw'] = (
            group["load_mw"]
            .interpolate(method='linear')
            .ffill()
            .bfill()
        )
        
        cleaned_parts.append(group)
        
        out = pd.concat(cleaned_parts).reset_index()
        
        out = out.sort_values(["region", "timestamp"]).reset_index(drop=True)

        
        return out

def validate(df):
    columns = df.columns.tolist()
    required_columns = ['timestamp', 'load_mw', 'region']
    missing = set(required_columns) - set(columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'],errors='raise')
    df['load_mw'] = df["load_mw"].astype(float)
    df['region'] = df['region'].astype('string')
    if df['timestamp'].isnull().any():
        raise ValueError("Null timestamps detected - schema validation failed.")
    
    df =df.sort_values(by='timestamp').reset_index(drop=True)
    
    return df
    
