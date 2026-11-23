#include "mlengine/math/Matrix.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace veloce;

void run_matrix_tests() {
    std::cout << "[TEST] Running Matrix tests...\n";

    // 1. Creation and indexing
    Matrix A = {{1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}};
    assert(A.rows() == 2);
    assert(A.cols() == 3);
    assert(std::abs(A(0, 1) - 2.0) < 1e-9);
    assert(std::abs(A(1, 2) - 6.0) < 1e-9);

    // 2. Transpose
    Matrix At = A.transpose();
    assert(At.rows() == 3);
    assert(At.cols() == 2);
    assert(std::abs(At(1, 0) - 2.0) < 1e-9);
    assert(std::abs(At(2, 1) - 6.0) < 1e-9);

    // 3. Dot Product: [2x3] * [3x2] -> [2x2]
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 1], [2, 3]]
    // Expected C = [[1*7+2*9+3*2, 1*8+2*1+3*3], [4*7+5*9+6*2, 4*8+5*1+6*3]]
    //            = [[7+18+6, 8+2+9], [28+45+12, 32+5+18]]
    //            = [[31, 19], [85, 55]]
    Matrix B = {{7.0, 8.0},
                {9.0, 1.0},
                {2.0, 3.0}};
    Matrix C = A.dot(B);
    assert(C.rows() == 2);
    assert(C.cols() == 2);
    assert(std::abs(C(0, 0) - 31.0) < 1e-9);
    assert(std::abs(C(0, 1) - 19.0) < 1e-9);
    assert(std::abs(C(1, 0) - 85.0) < 1e-9);
    assert(std::abs(C(1, 1) - 55.0) < 1e-9);

    // 4. Element-wise arithmetic
    Matrix D = A + A;
    assert(std::abs(D(0, 0) - 2.0) < 1e-9);
    assert(std::abs(D(1, 2) - 12.0) < 1e-9);

    Matrix E = A * 2.0;
    assert(std::abs(E(0, 0) - 2.0) < 1e-9);
    assert(std::abs(E(1, 2) - 12.0) < 1e-9);

    // 5. Numerically stable Sigmoid
    Matrix Z = {{-1000.0, 0.0, 1000.0}};
    Matrix S = Z.sigmoid();
    assert(S(0, 0) >= 0.0 && S(0, 0) < 1e-5);
    assert(std::abs(S(0, 1) - 0.5) < 1e-9);
    assert(S(0, 2) <= 1.0 && S(0, 2) > 0.9999);

    // 6. Bias column addition
    Matrix with_bias = A.add_bias_column(1.0);
    assert(with_bias.rows() == 2);
    assert(with_bias.cols() == 4);
    assert(std::abs(with_bias(0, 0) - 1.0) < 1e-9);
    assert(std::abs(with_bias(0, 1) - 1.0) < 1e-9);
    assert(std::abs(with_bias(1, 3) - 6.0) < 1e-9);

    // 7. Sum and Mean along axes
    Matrix sum0 = A.sum_axis(0); // sum columns across rows -> [5, 7, 9]
    assert(sum0.rows() == 1 && sum0.cols() == 3);
    assert(std::abs(sum0(0, 0) - 5.0) < 1e-9);
    assert(std::abs(sum0(0, 1) - 7.0) < 1e-9);
    assert(std::abs(sum0(0, 2) - 9.0) < 1e-9);

    Matrix sum1 = A.sum_axis(1); // sum rows across cols -> [6, 15]
    assert(sum1.rows() == 2 && sum1.cols() == 1);
    assert(std::abs(sum1(0, 0) - 6.0) < 1e-9);
    assert(std::abs(sum1(1, 0) - 15.0) < 1e-9);

    std::cout << "[PASS] Matrix linear algebra tests passed!\n";
}
