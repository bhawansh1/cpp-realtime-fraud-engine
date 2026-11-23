#pragma once

#include "mlengine/math/Matrix.hpp"
#include <string>
#include <vector>
#include <utility>

namespace veloce {

struct TrainTestSplit {
    Matrix X_train;
    Matrix y_train;
    Matrix X_test;
    Matrix y_test;
};

class Dataset {
public:
    // Load CSV data into feature Matrix X and label Matrix y
    // label_col: column index of the target label (-1 means last column)
    // has_header: whether to skip the first row
    static std::pair<Matrix, Matrix> load_csv(const std::string& filepath,
                                              int label_col = -1,
                                              bool has_header = true);

    // Split dataset into train and test partitions
    static TrainTestSplit train_test_split(const Matrix& X,
                                          const Matrix& y,
                                          double test_ratio = 0.2,
                                          bool shuffle = true,
                                          unsigned int seed = 42);

    // Synthetic Fraud Dataset Generator (for benchmarking & testing)
    // Generates realistic credit card transactions with fraud probability
    static std::pair<Matrix, Matrix> generate_synthetic_fraud_dataset(size_t num_samples = 2000,
                                                                      size_t num_features = 6,
                                                                      double fraud_ratio = 0.15,
                                                                      unsigned int seed = 42);
};

class StandardScaler {
public:
    StandardScaler() = default;

    void fit(const Matrix& X);
    [[nodiscard]] Matrix transform(const Matrix& X) const;
    [[nodiscard]] Matrix fit_transform(const Matrix& X);
    [[nodiscard]] Matrix inverse_transform(const Matrix& X) const;

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);

    [[nodiscard]] const std::vector<double>& means() const noexcept { return means_; }
    [[nodiscard]] const std::vector<double>& stds() const noexcept { return stds_; }

private:
    std::vector<double> means_{};
    std::vector<double> stds_{};
    bool is_fitted_{false};
};

} // namespace veloce
