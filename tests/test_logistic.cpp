#include "mlengine/models/LogisticRegression.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace veloce;

void run_logistic_tests() {
    std::cout << "[TEST] Running Logistic Regression tests...\n";

    // Generate synthetic 2D binary classification data
    // Class 0: centered around (-2.0, -2.0)
    // Class 1: centered around ( 2.0,  2.0)
    const size_t N_per_class = 50;
    const size_t N = N_per_class * 2;
    Matrix X(N, 2);
    Matrix y(N, 1);

    std::mt19937 rng(1337);
    std::normal_distribution<double> dist_0(-2.0, 0.5);
    std::normal_distribution<double> dist_1(2.0, 0.5);

    for (size_t i = 0; i < N_per_class; ++i) {
        X(i, 0) = dist_0(rng);
        X(i, 1) = dist_0(rng);
        y(i, 0) = 0.0;
    }
    for (size_t i = 0; i < N_per_class; ++i) {
        size_t idx = N_per_class + i;
        X(idx, 0) = dist_1(rng);
        X(idx, 1) = dist_1(rng);
        y(idx, 0) = 1.0;
    }

    LogisticConfig cfg;
    cfg.learning_rate = 0.05;
    cfg.max_epochs = 80;
    cfg.batch_size = 16;
    cfg.use_adam = true;
    cfg.verbose = false;

    LogisticRegression model(cfg);
    model.fit(X, y);

    // Verify loss decreased
    assert(!model.loss_history().empty());
    double initial_loss = model.loss_history().front();
    double final_loss = model.loss_history().back();
    assert(final_loss < initial_loss);
    assert(final_loss < 0.20); // Should easily achieve low loss

    // Test predictions
    Matrix preds = model.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(preds(i, 0) - y(i, 0)) < 1e-5) {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / static_cast<double>(N);
    assert(accuracy >= 0.98); // Clean separable data should be >= 98%

    // Test serialization & deserialization
    std::string test_model_path = "test_logistic_model.bin";
    model.save(test_model_path);

    LogisticRegression loaded_model;
    loaded_model.load(test_model_path);

    Matrix loaded_preds = loaded_model.predict(X);
    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(preds(i, 0) - loaded_preds(i, 0)) < 1e-9);
    }

    std::cout << "[PASS] Logistic Regression tests passed (Accuracy: "
              << (accuracy * 100.0) << "%, Final Loss: " << final_loss << ")!\n";
}
