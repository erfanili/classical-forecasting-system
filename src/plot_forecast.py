# visualization/plot_forecast_timescales.py

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_forecast_timescales(
    df_merged: pd.DataFrame,
    region: str,
    model_name: str,
    output_dir: str = "plots"
) -> None:
    """
    Produce daily, monthly, and yearly actual vs forecast plots
    from a merged dataframe containing timestamp, region,
    load_mw (actual), and y_hat (forecast).
    """

    # Ensure plots directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Ensure timestamp is datetime ---
    if not pd.api.types.is_datetime64_any_dtype(df_merged["timestamp"]):
        df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"])

    # --- Filter to region ---
    df_region = df_merged[df_merged["region"] == region].copy()

    # --- Indexing ---
    df_region = df_region.sort_values("timestamp")
    df_region = df_region.set_index("timestamp")
    # breakpoint()
    # --- Resample: Daily, Monthly, Yearly ---
    df_daily = df_region[["load_mw", "y_hat"]].resample("D").mean()
    df_monthly = df_region[["load_mw", "y_hat"]].resample("M").mean()
    df_yearly = df_region[["load_mw", "y_hat"]].resample("A").mean()
    
    # breakpoint()
    # Helper for plotting
    def _plot(df_resampled, freq_label: str):
        plt.figure(figsize=(12, 6))
        plt.plot(df_resampled.index, df_resampled["load_mw"], label="Actual", linewidth=2)
        plt.plot(df_resampled.index, df_resampled["y_hat"], label="Forecast", linestyle="--", linewidth=2)

        plt.title(f"{model_name} — {region} — {freq_label} Actual vs Forecast")
        plt.xlabel(freq_label)
        plt.ylabel("Load (MW)")
        plt.legend()
        plt.grid(alpha=0.3)

        # Save to file
        fname = f"{region}_{model_name}_{freq_label.lower()}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Generate plots ---
    _plot(df_daily, "Daily")
    _plot(df_monthly, "Monthly")
    _plot(df_yearly, "Yearly")



import matplotlib.pyplot as plt


import os
import matplotlib.pyplot as plt

def joint_plot(train_pred, test_pred, title='plot'):
    df_plot = pd.concat([train_pred,test_pred])
    # Ensure output directory exists
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(14, 6))

    # Actual load
    plt.plot(df_plot.index, df_plot['load_mw'], label='Actual', color='black')

    # Train predictions
    if 'y_hat' in train_pred.columns:
        plt.scatter(train_pred.index, train_pred['y_hat'], label='Train Forecast', color='blue')

    # Test predictions
    if 'y_hat' in test_pred.columns:
        plt.scatter(test_pred.index, test_pred['y_hat'], label='Test Forecast', color='red')

    # Vertical line marking start of test
    if len(test_pred) > 0:
        plt.axvline(test_pred.index[0], color='green', linestyle='--', label='Test Start')

    # Title, labels, grid
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)

    # Improve formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Safe filename
    safe_title = title.replace("/", "_").replace(" ", "_")
    
    plt.savefig(f"plots/{safe_title}.jpg", dpi=150)
    plt.close()


def res_plot(train, test, pred_col = 'y_hat', title='plot'):
    # df_plot = pd.concat([train_pred,test_pred])
    # Ensure output directory exists
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(14, 6))

    # Actual load
    # plt.plot(df_plot.index, df_plot['load_mw'], label='Actual', color='black')

    # Train predictions
    if pred_col in train.columns:
        plt.plot(train.index, train[pred_col]-train['load_mw'], label='Train Forecast', color='blue')

    # Test predictions
    if pred_col in test.columns:
        plt.plot(test.index, test[pred_col]-test['load_mw'], label='Test Forecast', color='red')

    # Vertical line marking start of test
    if len(test) > 0:
        plt.axvline(test.index[0], color='green', linestyle='--', label='Test Start')

    # Title, labels, grid
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)

    # Improve formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Safe filename
    safe_title = title.replace("/", "_").replace(" ", "_")
    
    plt.savefig(f"plots/{safe_title}.jpg", dpi=150)
    plt.close()
