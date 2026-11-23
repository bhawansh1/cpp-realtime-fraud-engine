#pragma once

#include "mlengine/models/IModel.hpp"
#include <vector>

namespace veloce {

struct LogisticConfig {
    double learning_rate{0.01};
    size_t max_epochs{100};
    size_t batch_size{32};
    double l2_lambda{0.01};
    bool use_adam{true};
    double beta1{0.9};
    double beta2{0.999};
    double epsilon{1e-8};
    double decision_threshold{0.5};
    bool verbose{false};
};

class LogisticRegression : public IModel {
public:
    explicit LogisticRegression(LogisticConfig config = {});

    void fit(const Matrix& X, const Matrix& y) override;
    [[nodiscard]] Matrix predict(const Matrix& X) const override;
    [[nodiscard]] Matrix predict_proba(const Matrix& X) const override;

    void save(const std::string& filepath) const override;
    void load(const std::string& filepath) override;

    [[nodiscard]] std::string name() const override { return "LogisticRegression"; }

    // Inspectors & Diagnostics
    [[nodiscard]] const Matrix& weights() const noexcept { return weights_; }
    [[nodiscard]] const std::vector<double>& loss_history() const noexcept { return loss_history_; }
    [[nodiscard]] double compute_loss(const Matrix& X_bias, const Matrix& y) const;

    void set_weights(const Matrix& weights) { weights_ = weights; }

private:
    LogisticConfig config_;
    Matrix weights_; // Dimension: (num_features + 1, 1) where weights_(0, 0) is bias
    std::vector<double> loss_history_;
};

} // namespace veloce
