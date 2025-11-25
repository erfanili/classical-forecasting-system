#run_evaluation.py
import pandas as pd
from ingestion.ingest_raw import load_single_csv, load_all_regions
from ingestion.validate_schema import validate, clean

from feature_engineering.feature_pipeline import build_features, FeatureConfig
from models.classical.ets_model import train_and_forecast_ets
from backtesting.evaluate import evaluate_forecast

# --- Load and clean raw AEP data ---
df_raw = load_all_regions("data/raw")
df = clean(validate(df_raw))

# --- Feature engineering ---
config = FeatureConfig(
    lags=[1, 24, 168],
    rolling_windows=[24, 168],
    use_boxcox=True,
    use_stl=True
)
# 1. Build features
df_feat = build_features(df, config)

# 2. Select training data (up to some cutoff)
region = "AEP"
cutoff = df_feat[df_feat["region"] == region]["timestamp"].max() - pd.Timedelta(hours=48)
df_train = df_feat[(df_feat["region"] == region) & (df_feat["timestamp"] <= cutoff)]

# 3. Train + forecast
df_forecast = train_and_forecast_ets(
    df=df_train,
    region=region,
    seasonal_periods=24,
    horizon=48
)

# 4. Get actual future values to compare
df_eval = df[df["region"] == region]
df_actual = df_eval[df_eval["timestamp"].isin(df_forecast["timestamp"])]

# 5. Merge for evaluation
df_merged = df_actual.merge(df_forecast, on=["timestamp", "region"], how="inner")

# 6. Run evaluation
metrics = evaluate_forecast(df_merged, df_merged)


# --- Print results ---
print("ETS Evaluation for AEP:")
for k, v in metrics.items():
    print(f"{k.upper()}: {v:.2f}")
# computer reference scale

df_aep = df[df["region"] == "AEP"]
mean_load = df_aep["load_mw"].mean()
peak_load = df_aep["load_mw"].max()

print("\n--- Load Scale for AEP ---")
print(f"Average Load: {mean_load:.2f} MW")
print(f"Peak Load:    {peak_load:.2f} MW")

# Print scaled errors
print("\n--- Scaled Error Ratios ---")
print(f"MAE / Avg Load:  {metrics['mae'] / mean_load * 100:.2f}%")
print(f"RMSE / Avg Load: {metrics['rmse'] / mean_load * 100:.2f}%")