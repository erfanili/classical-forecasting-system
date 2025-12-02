# --- Imports ---
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from argparse import ArgumentParser
from src.plot_forecast import plot_forecast_timescales, joint_plot, res_plot
from src.ingest import validate, clean,load_single_csv,detect_spikes
from src.feature_pipeline import add_calendar_features
from src.evaluate import evaluate_forecast, evaluate

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import json
os.makedirs('results/linear_fourier',exist_ok=True)
def diff_hours(d1, d2):
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2)
    return int((d2 - d1) / pd.Timedelta(hours=1))


def main(args):
    region = args.region
    df = load_single_csv(region)
    df = detect_spikes(df)
    

    df = df.reset_index()             # timestamp becomes a column
    df = df.set_index('timestamp')

    df_feat = add_calendar_features(df).dropna()
    df_feat = df_feat.drop(columns=['timestamp'], errors='ignore')

    exclude = ['load_mw', 'region','date']
    # feature_cols = [c for c in df_feat.columns if c not in exclude]
    feature_cols = ['sin_1','cos_1','sin_2','cos_2','sin_dow','cos_dow','sin_doy','cos_doy']

    train_start,train_end = tuple([pd.to_datetime(x) for x in args.train_period.split(':')])
    test_start, test_end = tuple([pd.to_datetime(x) for x in args.test_period.split(':')])
    df_train = df_feat.loc[train_start:train_end]
    # breakpoint()
    df_test = df_feat.loc[test_start:test_end]
    X_train = df_train[feature_cols]
    y_train = df_train['load_mw']

    X_test = df_test[feature_cols]
    y_test = df_test['load_mw']


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    main_model = LinearRegression()
    main_model.fit(X_train_scaled, y_train)

    df_train['y_hat'] = main_model.predict(X_train_scaled)
    df_test['y_hat']  = main_model.predict(X_test_scaled)


    actual_load = df_test['load_mw']
    pred_load = df_test['y_hat']

    steps = diff_hours(test_start,test_end)

    plt.figure(figsize=(12,5))
    plt.plot(actual_load.index, actual_load.values, label="Actual")
    plt.plot(actual_load.index, pred_load, label="Predicted")
    plt.legend()
    plt.title(f"{steps}-hour Ahead Load Forecast")
    
    plt.savefig(f'results/linear_fourier/test_lin_{test_start}.jpg')



    # --- 11. Evaluation ---
    results = evaluate(
        actual=df_test['load_mw'],
        pred=df_test['y_hat']
    )
    
    with open(f'results/linear_fourier/test_lin_{test_start}.json','w') as f:
        json.dump(results,f)
        
    return results

    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--region',default='PJME')
    parser.add_argument('--window',type = int,default=168)
    parser.add_argument('--train_period',default='2008-01-01:2011-09-13')
    parser.add_argument('--test_period',default='2011-09-13:2011-09-16')
    
    args = parser.parse_args()
    
    results = main(args)
    print(results)