#ingestion/ingest_raw.py
import pandas as pd 
from pathlib import Path
import numpy as np

def load_single_csv(region):
    file_path= Path(f'data/raw/{region}_hourly.csv')
    if file_path.is_file():
        
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Datetime':'timestamp', f'{region}_MW': 'load_mw'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        df = df.set_index('timestamp')
        df['region'] = region
        return df[["load_mw", "region"]]
    else:
        raise FileExistsError(f'No data for region {region}.')

def load_all_regions(path):
    folder = Path(path)
    csv_files  = [f for f in folder.glob("*_hourly.csv") if f.name != "PJM_Load_hourlt.csv"]
    csv_files = sorted(csv_files)
    
    if not csv_files:
        raise FileNotFoundError(f"No _hourly.csv files in {path}")
    
    
    frames = []
    for file_path in csv_files:
        df = load_single_csv(file_path)
        frames.append(df)
        
        
    full_df = pd.concat(frames,ignore_index=True)

    full_df = full_df.sort_values(["timestamp","region"]).reset_index(drop=True)
    
    
    return full_df



def detect_spikes(df, z_thresh = 3):
    
    df = df.copy()
    df['load_mw'] = df['load_mw'].astype(float)
    df['is_anomaly'] = False
    

    diff = df['load_mw'].diff()
    mean = diff.mean()
    std = diff.std()
        
    if std == 0 or np.isnan(std):
        z = pd.Series(0, index=diff.index)
    else:
        z = (diff - mean) / std
            
    anomalies = z.abs() > z_thresh
        
    df['is_anomaly']= anomalies
        
    return df
        
        


def clean(df):
    df = df.drop_duplicates(subset=['timestamp', 'region'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')

    cleaned_parts = []

    for region, group in df.groupby("region"):
        # region-specific timeline
        region_index = pd.date_range(
            start=group["timestamp"].min(),
            end=group["timestamp"].max(),
            freq="h"
        )

        group = group.sort_values("timestamp").set_index("timestamp")
        group = group.reindex(region_index)

        group.index.name = "timestamp"
        group["region"] = region

        group["load_mw"] = (
            group["load_mw"]
            .interpolate(method="linear")
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
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'],errors='raise')
    df['load_mw'] = df["load_mw"].astype(float)
    df['region'] = df['region'].astype('string')
    if df['timestamp'].isnull().any():
        raise ValueError("Null timestamps detected - schema validation failed.")
    
    df =df.sort_values(by='timestamp').reset_index(drop=True)
    
    return df
    

def save_cleaned_data(df,path):
    
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    
    try:
        df.to_parquet(output_path,index =False)
    except Exception as e:
        raise RuntimeError(f"Failed to write parquet file to {path}: {e}")
    
    
    
if __name__ == "__main__":

    RAW_PATH = 'data/raw'
    CLEAN_PATH = 'data/cleaned/clean_data.parquet'
    df = load_all_regions(path=RAW_PATH)
    df = validate(df)
    df = clean(df)
    df = detect_spikes(df)
    df = save_cleaned_data(df,path=CLEAN_PATH)