#backtesting/evaluate.py
import pandas as pd
import numpy as np

def evaluate_forecast(df, target_col="load_mw", pred_col="y_hat"):

    # merge rolling std into df and drop NaNs automatically
    df2 = df.copy()
    df2["std"] = df2[target_col].rolling(24).std()
    df2 = df2.dropna(subset=["std"])

    # extract aligned arrays
    actual = df2[target_col]
    pred = df2[pred_col]
    std = df2["std"]

    mae = (actual - pred).abs().mean()
    rmae = ((actual - pred).abs() / std).mean()

    return {"mae": mae, "rmae": rmae}


def evaluate(actual, pred):


    mae = (actual - pred).abs().mean()
    rmae = ((actual - pred).abs() / actual.mean()).mean()

    return {"mae": mae, "rmae": rmae}
