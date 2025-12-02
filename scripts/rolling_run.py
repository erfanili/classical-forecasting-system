import os
import json
import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import argparse
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.ingest import load_single_csv


# -------------------------------------------------------------------
# ROLLING EVALUATION FUNCTION (MODEL AGNOSTIC)
# -------------------------------------------------------------------

def run_rolling(model_fn, model_name, region="PJME", window=168,
                train_years=3, test_days=3, n_runs=20, year=2011):
    
    # --- output dir ---
    result_dir = f"results/{model_name}_rolling"
    os.makedirs(result_dir, exist_ok=True)

    # --- load data ---
    df = load_single_csv(region).sort_index()

    year_start = pd.Timestamp(f"{year}-01-01")
    year_end   = pd.Timestamp(f"{year}-12-31")

    total_days = (year_end - year_start).days
    step_days = total_days // n_runs

    print(f"\nRolling evaluation for model: {model_name}")
    print(f"Year: {year} | Step: {step_days} days\n")

    all_results = []

    current_test_start = year_start

    # ---------------------------------------------------------
    # RUN ALL ROLLING WINDOWS
    # ---------------------------------------------------------
    for i in range(n_runs):

        test_start = current_test_start
        test_end = test_start + timedelta(days=test_days)

        train_end = test_start
        train_start = train_end - relativedelta(years=train_years)

        print("====================================")
        print(f" RUN {i+1}/{n_runs} for model {model_name}")
        print("====================================")
        print(f"Train: {train_start.date()} → {train_end.date()}")
        print(f"Test : {test_start.date()} → {test_end.date()}")

        # call the model's main()
        args = argparse.Namespace(
            region=region,
            window=window,
            train_period=f"{train_start.date()}:{train_end.date()}",
            test_period=f"{test_start.date()}:{test_end.date()}",
        )
        res = model_fn(args)

        # store run metadata + result
        all_results.append({
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            **res
        })

        # move forward
        current_test_start += timedelta(days=step_days)

    # ---------------------------------------------------------
    # SAVE ALL RESULTS
    # ---------------------------------------------------------
    results_path = os.path.join(result_dir, "rolling_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved rolling results to: {results_path}")

    # ---------------------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------------------
    df_results = pd.DataFrame(all_results)

    summary = {
        "mean": df_results.mean(numeric_only=True).to_dict(),
        "std": df_results.std(numeric_only=True).to_dict()
    }

    summary_path = os.path.join(result_dir, "rolling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved summary to: {summary_path}")
    print("Done.\n")


# -------------------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------------------

if __name__ == "__main__":

    from pipelines.mlp_res24_pipeline import main as mlp_res24
    from pipelines.ets_pipeline import main as ets
    from pipelines.linear_fourier_pipeline import main as fourier
    from pipelines.naive24 import main as naive24
    # Example: run rolling evaluation for the MLP model
    run_rolling(
        model_fn=naive24,
        model_name="naive24",
        region="PJME",
        window=168,
        train_years=3,
        test_days=3,
        n_runs=20,
        year=2011
    )
