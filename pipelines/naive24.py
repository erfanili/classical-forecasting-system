# --- Imports ---
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from argparse import ArgumentParser
import json
from src.ingest import load_single_csv, detect_spikes
from src.evaluate import evaluate

import matplotlib.pyplot as plt

os.makedirs('results/naive24', exist_ok=True)


def main(args):
    region = args.region

    # ---------------- LOAD + CLEAN ----------------
    df = load_single_csv(region)
    df = detect_spikes(df)

    df = df.reset_index()              # timestamp becomes a column
    df = df.set_index('timestamp')     # ensure datetime index

    # ---------------- TRAIN/TEST SPLIT ----------------
    train_start, train_end = tuple(pd.to_datetime(x) for x in args.train_period.split(':'))
    test_start,  test_end  = tuple(pd.to_datetime(x) for x in args.test_period.split(':'))

    df_train = df.loc[train_start:train_end]
    df_test  = df.loc[test_start:test_end]

    # ---------------- NAIVE-24 FORECAST ----------------
    # y_hat(t) = y(t-24)
    df_test['y_hat'] = df['load_mw'].shift(24).loc[df_test.index]

    # if any missing (e.g. first 24 hours), drop them
    df_test = df_test.dropna(subset=['y_hat'])

    # ---------------- PLOT ----------------
    plt.figure(figsize=(12,5))
    plt.plot(df_test.index, df_test['load_mw'], label="Actual")
    plt.plot(df_test.index, df_test['y_hat'], label="Naive-24 Predicted")
    plt.legend()
    plt.title(f"Naive-24 Forecast ({region})")
    plt.savefig("results/naive24/test_naive24.jpg")

    # ---------------- EVALUATION ----------------
    results = evaluate(
        actual=df_test['load_mw'],
        pred=df_test['y_hat']
    )
    
    with open(f'results/naive24/test_naive24.json','w') as f:
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
