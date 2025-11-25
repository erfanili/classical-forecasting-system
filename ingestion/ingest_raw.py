import pandas as pd 
from pathlib import Path


def load_single_csv(file_path):
    file_path= Path(file_path)
    region = file_path.stem.split("_")[0]
    print(region)
    df = pd.read_csv(file_path)
    # df = df.drop(df.columns[0], axis=1)
    column_names = df.columns.tolist()
    df.rename(columns = {"Datetime": "timestamp",column_names[1]:"load_mw"}, inplace=True)
    df['region'] = region
    return df[["timestamp", "load_mw", "region"]]


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


