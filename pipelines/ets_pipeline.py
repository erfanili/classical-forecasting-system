# --- Imports ---
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from argparse import ArgumentParser

from src.ingest import load_single_csv, detect_spikes
from src.evaluate import evaluate
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import matplotlib.pyplot as plt
import json
os.makedirs('results/ets', exist_ok=True)


def diff_hours(d1, d2):
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2)
    return int((d2 - d1) / pd.Timedelta(hours=1))


def main(args):
    region = args.region

    # ---------------- LOAD + CLEAN ----------------
    df = load_single_csv(region)
    df = detect_spikes(df)

    df = df.reset_index()              # timestamp becomes a column
    df = df.set_index('timestamp')     # enforce datetime index

    # ---------------- TRAIN/TEST SPLIT ----------------
    train_start, train_end = tuple(pd.to_datetime(x) for x in args.train_period.split(':'))
    test_start,  test_end  = tuple(pd.to_datetime(x) for x in args.test_period.split(':'))

    df_train = df.loc[train_start:train_end]
    df_test  = df.loc[test_start:test_end]

    y_train = df_train['load_mw']
    y_test  = df_test['load_mw']

    # ---------------- FIT ETS MODEL ----------------
    # Seasonal period = 24 hours (daily cycle)
    model = ExponentialSmoothing(
        y_train,
        trend="add",
        seasonal="add",
        seasonal_periods=24
    ).fit(optimized=True)

    steps = len(y_test)
    y_hat = model.forecast(steps)

    df_test['y_hat'] = y_hat.values

    # ---------------- PLOT ----------------
    plt.figure(figsize=(12, 5))
    plt.plot(df_test.index, df_test['load_mw'], label="Actual")
    plt.plot(df_test.index, df_test['y_hat'], label="ETS Predicted")
    plt.legend()
    plt.title(f"ETS Forecast ({steps} hours)")
    plt.savefig("results/ets/test_ets.jpg")

    # ---------------- EVALUATION ----------------
    results = evaluate(
        actual=df_test['load_mw'],
        pred=df_test['y_hat']
    )
    
    with open(f'results/ets/test_ets.json','w') as f:
        json.dump(results,f)
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--region', default='PJME')
    parser.add_argument('--train_period', default='2008-01-01:2011-09-01')
    parser.add_argument('--test_period',  default='2011-09-01:2011-09-04')

    args = parser.parse_args()

    results = main(args)
    print(results)
