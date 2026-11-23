#include "mlengine/models/LogisticRegression.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

namespace veloce {

LogisticRegression::LogisticRegression(LogisticConfig config)
    : config_(config), weights_(), loss_history_() {}

double LogisticRegression::compute_loss(const Matrix& X_bias, const Matrix& y) const {
    const size_t N = X_bias.rows();
    if (N == 0) return 0.0;

    Matrix z = X_bias.dot(weights_);
    Matrix y_hat = z.sigmoid();

    double total_bce = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double p = std::clamp(y_hat(i, 0), 1e-15, 1.0 - 1e-15);
        double target = y(i, 0);
        total_bce += target * std::log(p) + (1.0 - target) * std::log(1.0 - p);
    }
    double loss = -total_bce / static_cast<double>(N);

    // L2 regularization term (ignore bias at index 0)
    double l2_sum = 0.0;
    for (size_t j = 1; j < weights_.rows(); ++j) {
        l2_sum += weights_(j, 0) * weights_(j, 0);
    }
    loss += (config_.l2_lambda / (2.0 * static_cast<double>(N))) * l2_sum;

    return loss;
}

void LogisticRegression::fit(const Matrix& X, const Matrix& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("Features X and labels y must have same number of rows");
    }
    if (y.cols() != 1) {
        throw std::invalid_argument("Target y must be a single column vector (Nx1)");
    }

    const size_t N = X.rows();
    const size_t D = X.cols();

    Matrix X_bias = X.add_bias_column(1.0); // Shape: N x (D + 1)

    // Xavier initialization for weights
    weights_ = Matrix::xavier_init(D + 1, 1);

    // Adam optimizer momentum variables
    Matrix m = Matrix::zeros(D + 1, 1);
    Matrix v = Matrix::zeros(D + 1, 1);
    size_t timestep = 0;

    size_t batch_size = config_.batch_size == 0 ? N : std::min(config_.batch_size, N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    loss_history_.clear();
    loss_history_.reserve(config_.max_epochs);

    for (size_t epoch = 0; epoch < config_.max_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t offset = 0; offset < N; offset += batch_size) {
            size_t current_batch_size = std::min(batch_size, N - offset);

            // Construct mini-batch
            Matrix X_b(current_batch_size, D + 1);
            Matrix y_b(current_batch_size, 1);
            for (size_t i = 0; i < current_batch_size; ++i) {
                size_t idx = indices[offset + i];
                for (size_t c = 0; c < D + 1; ++c) {
                    X_b(i, c) = X_bias(idx, c);
                }
                y_b(i, 0) = y(idx, 0);
            }

            // Forward pass: Z = X_b * W, y_hat = sigmoid(Z)
            Matrix Z = X_b.dot(weights_);
            Matrix y_hat = Z.sigmoid();

            // Gradient: G = (1 / B) * X_b^T * (y_hat - y_b)
            Matrix error = y_hat - y_b;
            Matrix grad = X_b.transpose().dot(error) * (1.0 / static_cast<double>(current_batch_size));

            // L2 regularization gradient (don't penalize bias term at index 0)
            for (size_t j = 1; j < D + 1; ++j) {
                grad(j, 0) += (config_.l2_lambda / static_cast<double>(current_batch_size)) * weights_(j, 0);
            }

            // Weight update
            if (config_.use_adam) {
                timestep++;
                double lr_t = config_.learning_rate *
                              std::sqrt(1.0 - std::pow(config_.beta2, static_cast<double>(timestep))) /
                              (1.0 - std::pow(config_.beta1, static_cast<double>(timestep)));

                for (size_t j = 0; j < D + 1; ++j) {
                    double g = grad(j, 0);
                    m(j, 0) = config_.beta1 * m(j, 0) + (1.0 - config_.beta1) * g;
                    v(j, 0) = config_.beta2 * v(j, 0) + (1.0 - config_.beta2) * (g * g);
                    weights_(j, 0) -= lr_t * m(j, 0) / (std::sqrt(v(j, 0)) + config_.epsilon);
                }
            } else {
                // Vanilla Mini-batch SGD
                weights_ -= grad * config_.learning_rate;
            }
        }

        double epoch_loss = compute_loss(X_bias, y);
        loss_history_.push_back(epoch_loss);

        if (config_.verbose && (epoch % 10 == 0 || epoch == config_.max_epochs - 1)) {
            std::cout << "[LogisticRegression] Epoch " << epoch << "/" << config_.max_epochs
                      << " - Loss: " << epoch_loss << "\n";
        }
    }
}

Matrix LogisticRegression::predict_proba(const Matrix& X) const {
    if (weights_.empty()) {
        throw std::runtime_error("Cannot predict with untrained LogisticRegression model.");
    }
    Matrix X_bias = X.add_bias_column(1.0);
    Matrix z = X_bias.dot(weights_);
    return z.sigmoid();
}

Matrix LogisticRegression::predict(const Matrix& X) const {
    Matrix probas = predict_proba(X);
    Matrix preds(probas.rows(), 1);
    for (size_t i = 0; i < probas.rows(); ++i) {
        preds(i, 0) = (probas(i, 0) >= config_.decision_threshold) ? 1.0 : 0.0;
    }
    return preds;
}

void LogisticRegression::save(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for saving model: " + filepath);
    }
    out << "VELOCEML_LOGISTIC_REGRESSION_V1\n";
    out << weights_.rows() << " " << weights_.cols() << "\n";
    for (size_t i = 0; i < weights_.rows(); ++i) {
        out << weights_(i, 0) << "\n";
    }
    out << config_.decision_threshold << "\n";
}

void LogisticRegression::load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for loading model: " + filepath);
    }
    std::string header;
    in >> header;
    if (header != "VELOCEML_LOGISTIC_REGRESSION_V1") {
        throw std::runtime_error("Invalid model format or unsupported header: " + header);
    }
    size_t rows, cols;
    in >> rows >> cols;
    weights_ = Matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        in >> weights_(i, 0);
    }
    in >> config_.decision_threshold;
}

} // namespace veloce
