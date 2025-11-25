#ets_model.py
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from pathlib import Path
def fit_ets_model(df, region, seasonal_periods):
    
    df_region = df[df["region"]== region].copy()
    
    if df_region.empty:
        
        raise ValueError(f"No data found for region: {region}")
    

    df_region = df_region.sort_values("timestamp")
    
    
    y = (
    df_region[["timestamp", "load_mw"]]
    .set_index("timestamp")
    .sort_index()["load_mw"]
    .astype(float)
)
    
    model = ExponentialSmoothing(
        y,
        trend = "add",
        seasonal="add",
        seasonal_periods = seasonal_periods,
        initialization_method = "estimated",
    )
    
    fitted_model = model.fit()
    fitted_model.region = region
    return fitted_model



def forecast_ets_model(model, horizon):
    y_hat = model.forecast(horizon)
    
    region = getattr(model, "region", None)
    if region is None:
        
        region = "unknown_region"
        
    last_timestamp = model.model.data.row_labels[-1]
    if not isinstance(last_timestamp, pd.Timestamp):
        last_timestamp = pd.to_datetime(last_timestamp)
        
    future_timestamps = pd.date_range(
        start= last_timestamp + pd.Timedelta(hours=1),
        periods = horizon,
        freq='h'        
    )
    
    df_fcst = pd.DataFrame({
        "timestamp": future_timestamps,
        "region": region,
        "y_hat": y_hat.values
    })
    
    return df_fcst


def train_and_forecast_ets(df, region, seasonal_periods, horizon):
    fitted_model  = fit_ets_model(df=df, region=region, seasonal_periods=seasonal_periods)
    fcst_df = forecast_ets_model(fitted_model, horizon)
    return fcst_df


# ---------------------------------------------------------
# 2. Save ETS Model
# ---------------------------------------------------------

def save_ets_model(model, path: str) -> None:
    """
    Save the ETS fitted model to disk using joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, path)
    except Exception as e:
        raise RuntimeError(f"Failed to save ETS model to {path}: {e}")


# ---------------------------------------------------------
# 3. Load ETS Model
# ---------------------------------------------------------

def load_ets_model(path: str):
    """
    Load a previously saved ETS model using joblib.
    Returns a statsmodels FittedModel.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"ETS model file not found: {path}")

    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load ETS model from {path}: {e}")