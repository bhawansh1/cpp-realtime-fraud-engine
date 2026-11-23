#include "mlengine/data/Dataset.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

namespace veloce {

std::pair<Matrix, Matrix> Dataset::load_csv(const std::string& filepath, int label_col, bool has_header) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + filepath);
    }

    std::string line;
    if (has_header && std::getline(file, line)) {
        // Skipped header row
    }

    std::vector<std::vector<double>> raw_rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row_vals;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            if (!token.empty()) {
                row_vals.push_back(std::stod(token));
            }
        }
        if (!row_vals.empty()) {
            raw_rows.push_back(std::move(row_vals));
        }
    }

    if (raw_rows.empty()) {
        throw std::runtime_error("CSV file contains no valid data rows: " + filepath);
    }

    size_t num_rows = raw_rows.size();
    size_t total_cols = raw_rows[0].size();
    size_t target_idx = (label_col < 0) ? (total_cols + label_col) : static_cast<size_t>(label_col);

    if (target_idx >= total_cols) {
        throw std::out_of_range("Label column index out of range in CSV.");
    }

    size_t num_features = total_cols - 1;
    Matrix X(num_rows, num_features);
    Matrix y(num_rows, 1);

    for (size_t r = 0; r < num_rows; ++r) {
        size_t feat_col = 0;
        for (size_t c = 0; c < total_cols; ++c) {
            if (c == target_idx) {
                y(r, 0) = raw_rows[r][c];
            } else {
                X(r, feat_col++) = raw_rows[r][c];
            }
        }
    }

    return {X, y};
}

TrainTestSplit Dataset::train_test_split(const Matrix& X, const Matrix& y, double test_ratio, bool shuffle, unsigned int seed) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have same row count for train_test_split.");
    }

    const size_t N = X.rows();
    const size_t D = X.cols();
    size_t test_size = static_cast<size_t>(static_cast<double>(N) * test_ratio);
    size_t train_size = N - test_size;

    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    Matrix X_train(train_size, D);
    Matrix y_train(train_size, 1);
    Matrix X_test(test_size, D);
    Matrix y_test(test_size, 1);

    for (size_t i = 0; i < train_size; ++i) {
        size_t idx = indices[i];
        for (size_t c = 0; c < D; ++c) {
            X_train(i, c) = X(idx, c);
        }
        y_train(i, 0) = y(idx, 0);
    }

    for (size_t i = 0; i < test_size; ++i) {
        size_t idx = indices[train_size + i];
        for (size_t c = 0; c < D; ++c) {
            X_test(i, c) = X(idx, c);
        }
        y_test(i, 0) = y(idx, 0);
    }

    return {X_train, y_train, X_test, y_test};
}

std::pair<Matrix, Matrix> Dataset::generate_synthetic_fraud_dataset(size_t num_samples,
                                                                    size_t num_features,
                                                                    double fraud_ratio,
                                                                    unsigned int seed) {
    Matrix X(num_samples, num_features);
    Matrix y(num_samples, 1);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u_dist(0.0, 1.0);

    // Normal transaction feature distributions
    std::lognormal_distribution<double> norm_amount(3.5, 0.7);     // ~$30 - $150
    std::exponential_distribution<double> norm_dist_home(0.05);   // ~20 km
    std::exponential_distribution<double> norm_dist_last(0.1);    // ~10 km
    std::poisson_distribution<int> norm_freq(3);                  // 3 tx / 24h
    std::uniform_real_distribution<double> norm_risk(0.05, 0.35); // Low device risk
    std::poisson_distribution<int> norm_retries(0);               // 0 retries

    // Fraud transaction distributions
    std::lognormal_distribution<double> fraud_amount(5.5, 1.0);    // ~$250 - $2000
    std::exponential_distribution<double> fraud_dist_home(0.005);  // ~200 km
    std::exponential_distribution<double> fraud_dist_last(0.01);   // ~100 km
    std::poisson_distribution<int> fraud_freq(14);                 // rapid burst
    std::uniform_real_distribution<double> fraud_risk(0.65, 0.98); // High device risk
    std::poisson_distribution<int> fraud_retries(2);              // Multiple PIN/OTP fails

    for (size_t i = 0; i < num_samples; ++i) {
        bool is_fraud = (u_dist(rng) < fraud_ratio);
        y(i, 0) = is_fraud ? 1.0 : 0.0;

        if (!is_fraud) {
            X(i, 0) = norm_amount(rng);
            X(i, 1) = norm_dist_home(rng);
            X(i, 2) = norm_dist_last(rng);
            X(i, 3) = static_cast<double>(norm_freq(rng));
            X(i, 4) = norm_risk(rng);
            X(i, 5) = static_cast<double>(norm_retries(rng));
        } else {
            X(i, 0) = fraud_amount(rng);
            X(i, 1) = fraud_dist_home(rng);
            X(i, 2) = fraud_dist_last(rng);
            X(i, 3) = static_cast<double>(fraud_freq(rng));
            X(i, 4) = fraud_risk(rng);
            X(i, 5) = static_cast<double>(fraud_retries(rng));
        }
    }

    return {X, y};
}

void StandardScaler::fit(const Matrix& X) {
    const size_t N = X.rows();
    const size_t D = X.cols();
    if (N == 0 || D == 0) return;

    means_.assign(D, 0.0);
    stds_.assign(D, 0.0);

    for (size_t c = 0; c < D; ++c) {
        double sum = 0.0;
        for (size_t r = 0; r < N; ++r) {
            sum += X(r, c);
        }
        means_[c] = sum / static_cast<double>(N);

        double sq_sum = 0.0;
        for (size_t r = 0; r < N; ++r) {
            double diff = X(r, c) - means_[c];
            sq_sum += diff * diff;
        }
        stds_[c] = std::sqrt(sq_sum / static_cast<double>(N));
        if (stds_[c] < 1e-10) {
            stds_[c] = 1.0; // avoid division by zero for constant features
        }
    }
    is_fitted_ = true;
}

Matrix StandardScaler::transform(const Matrix& X) const {
    if (!is_fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before transforming data.");
    }
    if (X.cols() != means_.size()) {
        throw std::invalid_argument("Column count mismatch in StandardScaler::transform.");
    }

    Matrix result(X.rows(), X.cols());
    for (size_t r = 0; r < X.rows(); ++r) {
        for (size_t c = 0; c < X.cols(); ++c) {
            result(r, c) = (X(r, c) - means_[c]) / stds_[c];
        }
    }
    return result;
}

Matrix StandardScaler::fit_transform(const Matrix& X) {
    fit(X);
    return transform(X);
}

Matrix StandardScaler::inverse_transform(const Matrix& X) const {
    if (!is_fitted_) {
        throw std::runtime_error("StandardScaler must be fitted before inverse transform.");
    }
    Matrix result(X.rows(), X.cols());
    for (size_t r = 0; r < X.rows(); ++r) {
        for (size_t c = 0; c < X.cols(); ++c) {
            result(r, c) = X(r, c) * stds_[c] + means_[c];
        }
    }
    return result;
}

void StandardScaler::save(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) throw std::runtime_error("Cannot save scaler to " + filepath);
    out << "VELOCEML_SCALER_V1\n";
    out << means_.size() << "\n";
    for (size_t i = 0; i < means_.size(); ++i) {
        out << means_[i] << " " << stds_[i] << "\n";
    }
}

void StandardScaler::load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) throw std::runtime_error("Cannot load scaler from " + filepath);
    std::string header;
    in >> header;
    if (header != "VELOCEML_SCALER_V1") throw std::runtime_error("Invalid scaler format: " + header);
    size_t size;
    in >> size;
    means_.resize(size);
    stds_.resize(size);
    for (size_t i = 0; i < size; ++i) {
        in >> means_[i] >> stds_[i];
    }
    is_fitted_ = true;
}

} // namespace veloce
