#evaluate.py
import pandas as pd
import numpy as np

def evaluate_forecast(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    target_col: str = "load_mw",
    pred_col: str = "y_hat"
) -> dict:
    """
    Evaluate forecast accuracy using MAE, RMSE, MAPE, and WAPE.
    
    df_actual must contain:   timestamp, region, load_mw
    df_forecast must contain: timestamp, region, y_hat
    """

    # ---- 1. Align actual + forecast on timestamp and region ----
    df = pd.merge(
        df_actual[["timestamp", "region", target_col]],
        df_forecast[["timestamp", "region", pred_col]],
        on=["timestamp", "region"],
        how="inner"
    )

    # Convert to arrays
    actual = df[target_col].astype(float).values
    pred = df[pred_col].astype(float).values

    # ---- 2. Metrics ----
    # MAE
    mae = np.mean(np.abs(actual - pred))

    # RMSE
    rmse = np.sqrt(np.mean((actual - pred) ** 2))

    # MAPE (avoid division by zero)
    mape = np.mean(
        np.abs((actual - pred) / np.maximum(actual, 1e-6))
    ) * 100

    # WAPE: Sum(|errors|) / Sum(actual)
    wape = (
        np.sum(np.abs(actual - pred)) /
        np.maximum(np.sum(actual), 1e-6)
    ) * 100


    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "wape": wape,
    }
