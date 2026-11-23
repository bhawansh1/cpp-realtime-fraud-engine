#!/usr/bin/env python3
"""
Python vs C++ Inference Benchmark Harness
Compares Scikit-Learn single-item prediction latency against VeloceML C++.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

def run_benchmark():
    print("========================================================")
    print("  Python (Scikit-Learn) Latency Benchmark Harness       ")
    print("========================================================\n")

    try:
        df = pd.read_csv("data/fraud_transactions.csv")
    except Exception as e:
        print(f"Error reading CSV: {e}. Please run veloce_train first.")
        return

    X = df.drop(columns=["is_fraud"]).values
    y = df["is_fraud"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Scikit-Learn models
    log_model = LogisticRegression(max_iter=200, solver="lbfgs")
    log_model.fit(X_scaled, y)

    tree_model = DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=4)
    tree_model.fit(X_scaled, y)

    NUM_SAMPLES = min(50000, len(X_scaled) * 20)
    indices = np.random.randint(0, len(X_scaled), size=NUM_SAMPLES)

    def benchmark_predict(model, name):
        # Warmup
        for i in range(500):
            _ = model.predict_proba(X_scaled[i % len(X_scaled)].reshape(1, -1))

        latencies = []
        t_start_total = time.perf_counter()
        for idx in indices:
            row = X_scaled[idx].reshape(1, -1)
            t0 = time.perf_counter()
            _ = model.predict_proba(row)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6) # microseconds
        t_end_total = time.perf_counter()

        latencies.sort()
        mean_us = np.mean(latencies)
        p50_us = np.percentile(latencies, 50)
        p95_us = np.percentile(latencies, 95)
        p99_us = np.percentile(latencies, 99)
        throughput = NUM_SAMPLES / (t_end_total - t_start_total)

        return {
            "name": name,
            "mean": mean_us,
            "p50": p50_us,
            "p95": p95_us,
            "p99": p99_us,
            "throughput": throughput
        }

    print(f"[*] Benchmarking {NUM_SAMPLES} single-item inference calls in Scikit-Learn...")
    log_res = benchmark_predict(log_model, "Scikit-Learn LogisticRegression")
    tree_res = benchmark_predict(tree_model, "Scikit-Learn DecisionTree")

    print("\n----------------------------------------------------------------------------")
    print("| Model (Python)           | Mean Latency |   p50 (Med)  |   p95   |   p99   | Throughput (ops/s) |")
    print("----------------------------------------------------------------------------")
    for r in [log_res, tree_res]:
        print(f"| {r['name']:<24} | {r['mean']:>8.2f} µs | {r['p50']:>8.2f} µs | {r['p95']:>5.2f} µs | {r['p99']:>5.2f} µs | {int(r['throughput']):>15} req/s |")
    print("----------------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_benchmark()
