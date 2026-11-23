#include "mlengine/metrics/Metrics.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace veloce {

double ClassificationMetrics::accuracy() const noexcept {
    size_t total = tp + tn + fp + fn;
    if (total == 0) return 0.0;
    return static_cast<double>(tp + tn) / static_cast<double>(total);
}

double ClassificationMetrics::precision() const noexcept {
    size_t denom = tp + fp;
    if (denom == 0) return 0.0;
    return static_cast<double>(tp) / static_cast<double>(denom);
}

double ClassificationMetrics::recall() const noexcept {
    size_t denom = tp + fn;
    if (denom == 0) return 0.0;
    return static_cast<double>(tp) / static_cast<double>(denom);
}

double ClassificationMetrics::f1_score() const noexcept {
    double p = precision();
    double r = recall();
    if (p + r < 1e-12) return 0.0;
    return 2.0 * (p * r) / (p + r);
}

ClassificationMetrics Metrics::evaluate(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.rows() != y_pred.rows()) {
        throw std::invalid_argument("Row count mismatch between y_true and y_pred");
    }

    ClassificationMetrics m;
    const size_t N = y_true.rows();
    for (size_t i = 0; i < N; ++i) {
        bool actual = (y_true(i, 0) >= 0.5);
        bool predicted = (y_pred(i, 0) >= 0.5);

        if (actual && predicted) m.tp++;
        else if (!actual && !predicted) m.tn++;
        else if (!actual && predicted) m.fp++;
        else if (actual && !predicted) m.fn++;
    }
    return m;
}

double Metrics::log_loss(const Matrix& y_true, const Matrix& y_proba) {
    if (y_true.rows() != y_proba.rows()) {
        throw std::invalid_argument("Row count mismatch between y_true and y_proba");
    }
    const size_t N = y_true.rows();
    if (N == 0) return 0.0;

    double total_loss = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double y = y_true(i, 0);
        double p = std::clamp(y_proba(i, 0), 1e-15, 1.0 - 1e-15);
        total_loss += y * std::log(p) + (1.0 - y) * std::log(1.0 - p);
    }
    return -total_loss / static_cast<double>(N);
}

void Metrics::print_report(const Matrix& y_true, const Matrix& y_pred, const std::string& model_name) {
    auto m = evaluate(y_true, y_pred);
    std::cout << "\n======================================================\n";
    std::cout << "          CLASSIFICATION REPORT: " << model_name << "\n";
    std::cout << "======================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Accuracy : " << std::setw(8) << (m.accuracy() * 100.0) << " %\n";
    std::cout << "  Precision: " << std::setw(8) << (m.precision() * 100.0) << " %\n";
    std::cout << "  Recall   : " << std::setw(8) << (m.recall() * 100.0) << " %\n";
    std::cout << "  F1 Score : " << std::setw(8) << (m.f1_score() * 100.0) << " %\n";
    std::cout << "------------------------------------------------------\n";
    std::cout << "  Confusion Matrix:\n";
    std::cout << "    True Positives  (TP): " << std::setw(6) << m.tp << " | False Positives (FP): " << std::setw(6) << m.fp << "\n";
    std::cout << "    False Negatives (FN): " << std::setw(6) << m.fn << " | True Negatives  (TN): " << std::setw(6) << m.tn << "\n";
    std::cout << "======================================================\n\n";
}

} // namespace veloce
