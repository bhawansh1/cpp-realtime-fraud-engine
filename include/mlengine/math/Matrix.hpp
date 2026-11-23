#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <string>
#include <stdexcept>
#include <random>
#include <cmath>
#include <initializer_list>

namespace veloce {

class Matrix {
public:
    // Lifecycle / Constructors
    Matrix();
    Matrix(size_t rows, size_t cols, double init_value = 0.0);
    Matrix(size_t rows, size_t cols, const std::vector<double>& data);
    Matrix(std::initializer_list<std::initializer_list<double>> init_list);

    // Rule of Five
    ~Matrix() = default;
    Matrix(const Matrix& other) = default;
    Matrix& operator=(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // Dimensions
    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    [[nodiscard]] const std::vector<double>& raw_data() const noexcept { return data_; }
    [[nodiscard]] std::vector<double>& raw_data() noexcept { return data_; }

    // Element Access
    [[nodiscard]] inline double& operator()(size_t r, size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    [[nodiscard]] inline const double& operator()(size_t r, size_t c) const noexcept {
        return data_[r * cols_ + c];
    }
    [[nodiscard]] double at(size_t r, size_t c) const;

    // Factories
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix eye(size_t n);
    static Matrix random_uniform(size_t rows, size_t cols, double low = 0.0, double high = 1.0, unsigned int seed = 42);
    static Matrix random_gaussian(size_t rows, size_t cols, double mean = 0.0, double stddev = 1.0, unsigned int seed = 42);
    static Matrix xavier_init(size_t rows, size_t cols, unsigned int seed = 42);

    // Matrix Arithmetic
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // Hadamard (element-wise) product
    Matrix operator/(const Matrix& other) const; // Element-wise division

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);

    // Scalar Arithmetic
    Matrix operator+(double scalar) const;
    Matrix operator-(double scalar) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix& operator+=(double scalar);
    Matrix& operator-=(double scalar);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    friend Matrix operator+(double scalar, const Matrix& mat);
    friend Matrix operator*(double scalar, const Matrix& mat);

    // Linear Algebra Operations
    [[nodiscard]] Matrix dot(const Matrix& other) const; // Matrix multiplication (optimized i-k-j)
    [[nodiscard]] Matrix transpose() const;

    // Slicing and Manipulation
    [[nodiscard]] Matrix get_row(size_t r) const;
    [[nodiscard]] Matrix get_col(size_t c) const;
    void set_row(size_t r, const Matrix& row_mat);
    void set_col(size_t c, const Matrix& col_mat);
    [[nodiscard]] Matrix slice_rows(size_t start_row, size_t end_row) const;
    [[nodiscard]] Matrix add_bias_column(double bias_value = 1.0) const;

    // Reductions
    [[nodiscard]] double sum() const noexcept;
    [[nodiscard]] double mean() const noexcept;
    [[nodiscard]] Matrix sum_axis(int axis) const;  // 0: across rows (returns 1xcols), 1: across cols (returns rowsx1)
    [[nodiscard]] Matrix mean_axis(int axis) const;

    // Mathematical Transformations
    [[nodiscard]] Matrix apply(const std::function<double(double)>& fn) const;
    void apply_inplace(const std::function<double(double)>& fn);

    [[nodiscard]] Matrix sigmoid() const;
    [[nodiscard]] Matrix clip(double min_val, double max_val) const;
    [[nodiscard]] Matrix log() const;
    [[nodiscard]] Matrix sqrt() const;

    // Utility
    void print(const std::string& name = "", size_t max_rows = 5, size_t max_cols = 5) const;

private:
    size_t rows_{0};
    size_t cols_{0};
    std::vector<double> data_{};
};

} // namespace veloce
