#include "mlengine/math/Matrix.hpp"
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace veloce {

Matrix::Matrix() : rows_(0), cols_(0), data_() {}

Matrix::Matrix(size_t rows, size_t cols, double init_value)
    : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match dimensions: expected " +
                                    std::to_string(rows * cols) + ", got " + std::to_string(data.size()));
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init_list) {
    rows_ = init_list.size();
    cols_ = rows_ > 0 ? init_list.begin()->size() : 0;
    data_.reserve(rows_ * cols_);
    for (const auto& row : init_list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows in initializer_list must have identical length.");
        }
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

double Matrix::at(size_t r, size_t c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix index out of bounds: (" + std::to_string(r) + ", " +
                                std::to_string(c) + ") for shape (" + std::to_string(rows_) + ", " +
                                std::to_string(cols_) + ")");
    }
    return data_[r * cols_ + c];
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::eye(size_t n) {
    Matrix m(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        m(i, i) = 1.0;
    }
    return m;
}

Matrix Matrix::random_uniform(size_t rows, size_t cols, double low, double high, unsigned int seed) {
    Matrix m(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(low, high);
    for (auto& val : m.data_) {
        val = dist(gen);
    }
    return m;
}

Matrix Matrix::random_gaussian(size_t rows, size_t cols, double mean, double stddev, unsigned int seed) {
    Matrix m(rows, cols);
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(mean, stddev);
    for (auto& val : m.data_) {
        val = dist(gen);
    }
    return m;
}

Matrix Matrix::xavier_init(size_t rows, size_t cols, unsigned int seed) {
    // Glorot / Xavier initialization: stddev = sqrt(2.0 / (fan_in + fan_out))
    double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
    return random_uniform(rows, cols, -limit, limit, seed);
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix addition shape mismatch: (" + std::to_string(rows_) + "x" +
                                    std::to_string(cols_) + ") vs (" + std::to_string(other.rows_) + "x" +
                                    std::to_string(other.cols_) + ")");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix subtraction shape mismatch: (" + std::to_string(rows_) + "x" +
                                    std::to_string(cols_) + ") vs (" + std::to_string(other.rows_) + "x" +
                                    std::to_string(other.cols_) + ")");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Hadamard product shape mismatch.");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Matrix Matrix::operator/(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Element-wise division shape mismatch.");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / (other.data_[i] + 1e-15);
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix += shape mismatch.");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix -= shape mismatch.");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix *= shape mismatch.");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= other.data_[i];
    }
    return *this;
}

Matrix Matrix::operator+(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Matrix Matrix::operator-(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - scalar;
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result(rows_, cols_);
    double inv = 1.0 / scalar;
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * inv;
    }
    return result;
}

Matrix& Matrix::operator+=(double scalar) {
    for (auto& val : data_) val += scalar;
    return *this;
}

Matrix& Matrix::operator-=(double scalar) {
    for (auto& val : data_) val -= scalar;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (auto& val : data_) val *= scalar;
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    double inv = 1.0 / scalar;
    for (auto& val : data_) val *= inv;
    return *this;
}

Matrix operator+(double scalar, const Matrix& mat) {
    return mat + scalar;
}

Matrix operator*(double scalar, const Matrix& mat) {
    return mat * scalar;
}

// Optimized i-k-j loop matrix multiplication (cache friendly)
Matrix Matrix::dot(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Dot product dimension mismatch: (" + std::to_string(rows_) + "x" +
                                    std::to_string(cols_) + ") * (" + std::to_string(other.rows_) + "x" +
                                    std::to_string(other.cols_) + ")");
    }

    Matrix C(rows_, other.cols_, 0.0);
    const size_t M = rows_;
    const size_t K = cols_;
    const size_t N = other.cols_;

    for (size_t i = 0; i < M; ++i) {
        const size_t a_row_offset = i * K;
        const size_t c_row_offset = i * N;
        for (size_t k = 0; k < K; ++k) {
            const double a_ik = data_[a_row_offset + k];
            const size_t b_row_offset = k * N;
            // Inner loop access is strictly contiguous in memory:
            // C[i][j] and other[k][j] both step sequentially along j
            for (size_t j = 0; j < N; ++j) {
                C.data_[c_row_offset + j] += a_ik * other.data_[b_row_offset + j];
            }
        }
    }
    return C;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return result;
}

Matrix Matrix::get_row(size_t r) const {
    if (r >= rows_) throw std::out_of_range("Row index out of range");
    std::vector<double> row_data(data_.begin() + (r * cols_), data_.begin() + ((r + 1) * cols_));
    return Matrix(1, cols_, row_data);
}

Matrix Matrix::get_col(size_t c) const {
    if (c >= cols_) throw std::out_of_range("Col index out of range");
    std::vector<double> col_data(rows_);
    for (size_t r = 0; r < rows_; ++r) {
        col_data[r] = (*this)(r, c);
    }
    return Matrix(rows_, 1, col_data);
}

void Matrix::set_row(size_t r, const Matrix& row_mat) {
    if (r >= rows_) throw std::out_of_range("Row index out of range");
    if (row_mat.cols_ != cols_) throw std::invalid_argument("Row width mismatch");
    for (size_t c = 0; c < cols_; ++c) {
        (*this)(r, c) = row_mat(0, c);
    }
}

void Matrix::set_col(size_t c, const Matrix& col_mat) {
    if (c >= cols_) throw std::out_of_range("Col index out of range");
    if (col_mat.rows_ != rows_) throw std::invalid_argument("Col height mismatch");
    for (size_t r = 0; r < rows_; ++r) {
        (*this)(r, c) = col_mat(r, 0);
    }
}

Matrix Matrix::slice_rows(size_t start_row, size_t end_row) const {
    if (start_row > end_row || end_row > rows_) {
        throw std::out_of_range("Invalid row slice range");
    }
    size_t new_rows = end_row - start_row;
    std::vector<double> slice_data(data_.begin() + (start_row * cols_),
                                  data_.begin() + (end_row * cols_));
    return Matrix(new_rows, cols_, slice_data);
}

Matrix Matrix::add_bias_column(double bias_value) const {
    Matrix result(rows_, cols_ + 1);
    for (size_t r = 0; r < rows_; ++r) {
        result(r, 0) = bias_value;
        for (size_t c = 0; c < cols_; ++c) {
            result(r, c + 1) = (*this)(r, c);
        }
    }
    return result;
}

double Matrix::sum() const noexcept {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Matrix::mean() const noexcept {
    if (data_.empty()) return 0.0;
    return sum() / static_cast<double>(data_.size());
}

Matrix Matrix::sum_axis(int axis) const {
    if (axis == 0) { // Collapse rows -> shape (1, cols)
        Matrix result(1, cols_, 0.0);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                result(0, c) += (*this)(r, c);
            }
        }
        return result;
    } else if (axis == 1) { // Collapse cols -> shape (rows, 1)
        Matrix result(rows_, 1, 0.0);
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                result(r, 0) += (*this)(r, c);
            }
        }
        return result;
    }
    throw std::invalid_argument("Axis must be 0 or 1");
}

Matrix Matrix::mean_axis(int axis) const {
    if (axis == 0) {
        Matrix s = sum_axis(0);
        return s / static_cast<double>(rows_);
    } else if (axis == 1) {
        Matrix s = sum_axis(1);
        return s / static_cast<double>(cols_);
    }
    throw std::invalid_argument("Axis must be 0 or 1");
}

Matrix Matrix::apply(const std::function<double(double)>& fn) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = fn(data_[i]);
    }
    return result;
}

void Matrix::apply_inplace(const std::function<double(double)>& fn) {
    for (auto& val : data_) {
        val = fn(val);
    }
}

Matrix Matrix::sigmoid() const {
    return apply([](double z) {
        // Numerically stable sigmoid function
        if (z >= 0.0) {
            return 1.0 / (1.0 + std::exp(-z));
        } else {
            double ez = std::exp(z);
            return ez / (1.0 + ez);
        }
    });
}

Matrix Matrix::clip(double min_val, double max_val) const {
    return apply([min_val, max_val](double v) {
        return std::clamp(v, min_val, max_val);
    });
}

Matrix Matrix::log() const {
    return apply([](double v) {
        return std::log(std::max(v, 1e-15));
    });
}

Matrix Matrix::sqrt() const {
    return apply([](double v) {
        return std::sqrt(std::max(v, 0.0));
    });
}

void Matrix::print(const std::string& name, size_t max_rows, size_t max_cols) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "Matrix (" << rows_ << "x" << cols_ << "):\n";
    size_t pr_rows = std::min(rows_, max_rows);
    size_t pr_cols = std::min(cols_, max_cols);

    for (size_t r = 0; r < pr_rows; ++r) {
        std::cout << "  [";
        for (size_t c = 0; c < pr_cols; ++c) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << (*this)(r, c) << " ";
        }
        if (cols_ > max_cols) std::cout << "...";
        std::cout << "]\n";
    }
    if (rows_ > max_rows) {
        std::cout << "  ... (" << rows_ - max_rows << " more rows)\n";
    }
}

} // namespace veloce
