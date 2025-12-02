#!/bin/bash

REGION="PJME"
WINDOW=168
SCRIPT="pipelines/mlp_res24_pipeline.py"

# Rolling parameters
TRAIN_YEARS=3
TEST_DAYS=3
N_RUNS=20

# Start date (must match your dataset start)
START_DATE="2008-01-01"

# Convert to GNU date-friendly format
current_test_start=$(date -I -d "$START_DATE +${TRAIN_YEARS} years")

for ((i=1; i<=N_RUNS; i++))
do
    echo "==========================="
    echo " Run $i / $N_RUNS"
    echo "==========================="

    # Compute rolling windows
    train_start=$(date -I -d "$current_test_start -${TRAIN_YEARS} years")
    train_end=$current_test_start
    test_start=$current_test_start
    test_end=$(date -I -d "$test_start +${TEST_DAYS} days")

    echo "Train: $train_start → $train_end"
    echo "Test : $test_start → $test_end"

    # Run your Python script
    python $SCRIPT \
        --region $REGION \
        --window $WINDOW \
        --train_period "${train_start}:${train_end}" \
        --test_period "${test_start}:${test_end}"

    # Move test_start forward
    current_test_start=$(date -I -d "$current_test_start +${TEST_DAYS} days")
done
