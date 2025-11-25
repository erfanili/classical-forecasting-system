from dataclasses import dataclass
import pandas as pd
from typing import List

from .calendar_features import add_calendar_features
from .lag_roll_features import add_rolling_features, add_lag_features
from .transforms import add_boxcox_transform
from .stl_decompostion import add_stl_components


@dataclass
class FeatureConfig:
    lags: List[int]
    rolling_windows: List[int]
    use_boxcox: bool=True
    use_stl: bool = True
    

def build_features(df, config):
    
    
    df = df.copy()
    
    df = add_calendar_features(df)
    
    df = add_lag_features(df,config.lags)
    
    df = add_rolling_features(df, config.rolling_windows)
    
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