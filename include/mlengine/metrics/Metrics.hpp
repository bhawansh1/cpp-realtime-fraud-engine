#pragma once

#include "mlengine/math/Matrix.hpp"
#include <string>

namespace veloce {

struct ClassificationMetrics {
    size_t tp{0};
    size_t tn{0};
    size_t fp{0};
    size_t fn{0};

    [[nodiscard]] double accuracy() const noexcept;
    [[nodiscard]] double precision() const noexcept;
    [[nodiscard]] double recall() const noexcept;
    [[nodiscard]] double f1_score() const noexcept;
};

class Metrics {
public:
    // Evaluate binary classification predictions against ground truth
    static ClassificationMetrics evaluate(const Matrix& y_true, const Matrix& y_pred);

    // Compute binary cross-entropy log loss on predicted probabilities
    static double log_loss(const Matrix& y_true, const Matrix& y_proba);

    // Pretty-printed classification report to terminal or log
    static void print_report(const Matrix& y_true,
                             const Matrix& y_pred,
                             const std::string& model_name = "Model");
};

} // namespace veloce
