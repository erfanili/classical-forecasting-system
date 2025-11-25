import pandas as pd
import numpy as np
import holidays



def add_calendar_features(df):
    
    
    df = df.copy()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['day_of_year'] = df['timestamp'].dt.day_of_year
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5,6])
    
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    us_holidays = holidays.US()
    
    df['date'] = df['timestamp'].dt.date
    df['is_holiday'] = df['date'].isin(us_holidays)
    
    df["is_pre_holiday"] = (df["date"] + pd.Timedelta(days=1)).isin(us_holidays)
    df["is_post_holiday"] = (df["date"] - pd.Timedelta(days=1)).isin(us_holidays)
    
    df.drop(columns = ['date'])
    
    
    return df