#feature_engineering/feature_pipeline.py
from dataclasses import dataclass
import pandas as pd
from typing import List




@dataclass
class FeatureConfig:
    lags: List[int]
    rolling_windows: List[int]
    use_boxcox: bool=True
    use_stl: bool = False
    
#feature_engineering/calendar_features.py
import pandas as pd
import numpy as np
import holidays



def add_calendar_features(df):
    
    def _make_fourier(df,k=2):
        df = df.copy()
        
        for k in range(1, k + 1):
            df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['hour']/ 24)
            df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['hour']/ 24)
        
        return df
    
    
    df = df.copy()
    df['timestamp'] = df.index
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['day_of_year'] = df['timestamp'].dt.day_of_year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5,6])
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] /365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] /365)
    df = _make_fourier(df, k=2)
    
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    us_holidays = holidays.US()
    
    df['date'] = df['timestamp'].dt.date
    df['is_holiday'] = df['date'].isin(us_holidays)
    
    df["is_pre_holiday"] = (df["date"] + pd.Timedelta(days=1)).isin(us_holidays)
    df["is_post_holiday"] = (df["date"] - pd.Timedelta(days=1)).isin(us_holidays)
    
    df.drop(columns = ['date'])
    
    
    return df




def add_lag_features(df, lags):
    df = df.copy()
    
    
    
    df = df.sort_values(['region', 'timestamp']).reset_index(drop=True)
    
    for k in lags:
        lag_col = f"lag_{k}"
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




def build_features(df, config):
    
    
    df = df.copy()
    
    df = add_calendar_features(df)
    
    
    if config.use_boxcox:
        df = add_boxcox_transform(df, per_region=True)
    
    if config.use_stl:
        df = add_stl_components(df)
        
    

    return df
    
    
if __name__ == "__main__":
    df_raw = pd.read_parquet("data/cleaned/clean_data.parquet")

    config = FeatureConfig(
        lags=[1, 24, 168],
        rolling_windows=[24, 168],
        use_boxcox=True,
        use_stl=True,
    )

    df_features = build_features(df_raw, config)
    print(df_features.head())