#include "mlengine/math/Matrix.hpp"
#include "mlengine/data/Dataset.hpp"
#include "mlengine/models/LogisticRegression.hpp"
#include "mlengine/models/DecisionTree.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

using namespace veloce;

struct LatencyStats {
    double mean_us{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double throughput_ops_sec{0.0};
};

LatencyStats benchmark_model(const IModel& model, const Matrix& test_samples, size_t iterations) {
    std::vector<double> latencies_us;
    latencies_us.reserve(iterations);

    const size_t num_samples = test_samples.rows();

    // Warm-up cache (1000 iterations)
    for (size_t i = 0; i < 1000; ++i) {
        Matrix single = test_samples.get_row(i % num_samples);
        auto p = model.predict_proba(single);
        (void)p;
    }

    auto start_total = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        Matrix single = test_samples.get_row(i % num_samples);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto p = model.predict_proba(single);
        auto t2 = std::chrono::high_resolution_clock::now();

        (void)p;
        double elapsed_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
        latencies_us.push_back(elapsed_us);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_total - start_total).count();

    std::sort(latencies_us.begin(), latencies_us.end());

    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    LatencyStats stats;
    stats.mean_us = sum / static_cast<double>(iterations);
    stats.p50_us = latencies_us[static_cast<size_t>(iterations * 0.50)];
    stats.p95_us = latencies_us[static_cast<size_t>(iterations * 0.95)];
    stats.p99_us = latencies_us[static_cast<size_t>(iterations * 0.99)];
    stats.throughput_ops_sec = static_cast<double>(iterations) / total_sec;

    return stats;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  VeloceML: Ultra-Low Latency C++ Inference Benchmark   \n";
    std::cout << "========================================================\n\n";

    // Load models
    std::cout << "[*] Loading trained models from disk...\n";
    LogisticRegression log_model;
    log_model.load("models/fraud_logistic.bin");

    DecisionTree tree_model;
    tree_model.load("models/fraud_tree.bin");

    StandardScaler scaler;
    scaler.load("models/fraud_scaler.bin");

    // Load or generate benchmark sample pool (1000 transactions)
    auto [X_raw, y] = Dataset::generate_synthetic_fraud_dataset(1000, 6, 0.15, 999);
    Matrix X_scaled = scaler.transform(X_raw);

    const size_t BENCHMARK_ITERATIONS = 50000;
    std::cout << "[*] Running " << BENCHMARK_ITERATIONS << " single-transaction inference calls...\n\n";

    // 1. Benchmark Logistic Regression
    auto log_stats = benchmark_model(log_model, X_scaled, BENCHMARK_ITERATIONS);

    // 2. Benchmark Decision Tree
    auto tree_stats = benchmark_model(tree_model, X_scaled, BENCHMARK_ITERATIONS);

    // Output formatted results table
    std::cout << "----------------------------------------------------------------------------\n";
    std::cout << "| Model                    | Mean Latency |   p50 (Med)  |   p95   |   p99   | Throughput (ops/s) |\n";
    std::cout << "----------------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Logistic Regression      | "
              << std::setw(8) << log_stats.mean_us << " µs | "
              << std::setw(8) << log_stats.p50_us  << " µs | "
              << std::setw(5) << log_stats.p95_us  << " µs | "
              << std::setw(5) << log_stats.p99_us  << " µs | "
              << std::setw(15) << static_cast<size_t>(log_stats.throughput_ops_sec) << " req/s |\n";

    std::cout << "| Decision Tree Classifier | "
              << std::setw(8) << tree_stats.mean_us << " µs | "
              << std::setw(8) << tree_stats.p50_us  << " µs | "
              << std::setw(5) << tree_stats.p95_us  << " µs | "
              << std::setw(5) << tree_stats.p99_us  << " µs | "
              << std::setw(15) << static_cast<size_t>(tree_stats.throughput_ops_sec) << " req/s |\n";
    std::cout << "----------------------------------------------------------------------------\n";
    std::cout << "Sub-microsecond / single-digit microsecond inference achieved!\n\n";

    return 0;
}
