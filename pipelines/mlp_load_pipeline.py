import sys
from pathlib import Path
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from src.ingest import (load_single_csv,
                        detect_spikes
)
from src.evaluate import evaluate
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import json

def build_res_dataset(series, window=168):

    values = series.values
    X, y = [], []

    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(values[i + window])  # predict next res24

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y



class MLPForecast(nn.Module):
    def __init__(self,window=48):
        super().__init__()
        self.window = window
        self.fc1 = nn.Linear(window,window)
        self.fc2 = nn.Linear(window,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
    


def forecast_res24_sequence(model, res_norm, window=48):
    """
    model: trained PyTorch model
    res_norm: normalized res24 series (pandas Series)
    returns: array of predicted normalized res24
    """
    model.eval()
    preds = []

    values = res_norm.values

    # generate predictions from window → end
    for i in range(window, len(values)):
        window_input = values[i-window:i]  # last 48 values
        x = torch.tensor(window_input, dtype=torch.float32).unsqueeze(0)  # shape (1, window)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)

    return np.array(preds)



def forecast_res_multistep(model, historical_data, steps: int):
    model.eval()
    preds = []

    # Make sure we're working with a 1D numpy array
    cur_window = np.asarray(historical_data, dtype=np.float32).copy()

    # Optional safety check
    assert cur_window.shape[0] == model.window, (
        f"Initial window length {cur_window.shape[0]} "
        f"does not match model.window={model.window}"
    )

    for _ in range(steps):
        x = torch.from_numpy(cur_window).float().unsqueeze(0)
  # shape (1, window)
        with torch.no_grad():
            pred_norm = model(x).item()
        preds.append(pred_norm)

        # Shift window: drop oldest, add predicted value
        cur_window = np.append(cur_window[1:], pred_norm)

    return np.array(preds, dtype=np.float32)


def reconstruct_load_from_res24(pred_res24, df_load, last_timestamp, steps):

    pred_load = []

    # Extract the last 24 actual load values
    last_24_actual = df_load.loc[
        last_timestamp - pd.Timedelta(hours=23) : last_timestamp
    ].values

    for k in range(steps):

        if k < 24:
            # Use ACTUAL lag-24 load values
            lag24_value = last_24_actual[k]
        else:
            # Use our OWN predicted load from k-24
            lag24_value = pred_load[k - 24]

        # Reconstruct load
        load_hat = pred_res24[k] + lag24_value
        pred_load.append(load_hat)

    return np.array(pred_load)

def add_hours(date_str, hours):
    return pd.to_datetime(date_str) + pd.Timedelta(hours=hours)

def diff_hours(d1, d2):
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2)
    return int((d2 - d1) / pd.Timedelta(hours=1))


def main(args):
    region = args.region
    df = load_single_csv(region)
    df = detect_spikes(df)
    

    df['res24'] = df['load_mw']
    df = df.dropna(subset=['res24'])


    res = df['res24'].dropna()
    train_start,train_end = tuple(args.train_period.split(':'))
    test_start, test_end = tuple(args.test_period.split(':'))
    
    train_start_padded = add_hours(train_start,-1*args.window+1) # add initial padding
    test_start_padded  = add_hours(test_start,-1*args.window+1)
    train_res = res.loc[train_start_padded:pd.to_datetime(train_end)]
    # Get last 168 valid residuals BEFORE test_start
    test_historical_data = res.loc[:pd.to_datetime(test_start)].tail(args.window)


    mean = train_res.mean()
    std = train_res.std()

    train_norm = (train_res - mean) / std
    test_historical_norm  = (test_historical_data - mean) / std

    # ---------------- FIX 1: correct training dataset ----------------
    X_train, y_train = build_res_dataset(train_norm, window=args.window)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                            batch_size=64, shuffle=True)

    model = MLPForecast(window=args.window).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    for epoch in range(20):
        for xb, yb in train_loader:
            xb = xb.cuda()
            yb = yb.cuda()
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())


    # ---------------- Forecasting ----------------

    steps = diff_hours(test_start,test_end)

    # breakpoint()
    
    model.to('cpu')
    pred_norm = forecast_res_multistep(model, test_historical_norm, steps)
    pred_load = pred_norm * std + mean

    # # Reconstruct load using proper lag-24 logic
    # pred_load = reconstruct_load_from_res24(
    #     pred_res24=pred_res24,
    #     df_load=df['load_mw'],
    #     last_timestamp=pd.to_datetime(test_start),
    #     steps=steps
    # )



    # ---------------- Actual future load ----------------
    # get test_start position
    idx_start = df.index.get_loc(pd.to_datetime(test_start))

    # next `steps` rows of actual load
    actual_load = df['load_mw'].iloc[idx_start+1 : idx_start+1+steps]
    
    results = evaluate(actual=actual_load,pred= pred_load)
    
    os.makedirs('results/mlp_load',exist_ok=True)
    with open(f'results/mlp_load/test_load_{args.window}.json','w') as f:
        json.dump(results,f)
    # ---------------- Plot ----------------
    plt.figure(figsize=(12,5))
    plt.plot(actual_load.index, actual_load.values, label="Actual")
    plt.plot(actual_load.index, pred_load, label="Predicted")
    plt.legend()
    plt.title(f"{steps}-hour Ahead Load Forecast")
    
    plt.savefig(f'results/mlp_load/test_load_{args.window}.jpg')














if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--region',default='PJME')
    parser.add_argument('--window',type = int,default=168)
    parser.add_argument('--train_period',default='2008-01-01:2011-09-01')
    parser.add_argument('--test_period',default='2011-09-01:2011-09-04')
    
    args = parser.parse_args()
    
    main(args)