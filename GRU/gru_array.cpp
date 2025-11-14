/**
 * @file gru_array.cpp
 * @brief C++ implementation of Gated Recurrent Unit (GRU) cell array
 */

#include "gru_array.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

namespace gru {

/* ============================================================================
 * Random Number Generation
 * ============================================================================ */

namespace {
    // Thread-safe random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    double randn() {
        return normal_dist(gen);
    }

    double sigmoid_func(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
}

/* ============================================================================
 * Matrix Implementation
 * ============================================================================ */

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, bool random)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
    if (random) {
        randomize();
    }
}

double& Matrix::operator()(size_t i, size_t j) {
    return data_[i * cols_ + j];
}

double Matrix::operator()(size_t i, size_t j) const {
    return data_[i * cols_ + j];
}

void Matrix::randomize() {
    for (auto& val : data_) {
        val = randn();
    }
}

std::vector<double> Matrix::operator*(const std::vector<double>& vec) const {
    if (vec.size() != cols_) {
        throw std::invalid_argument("Matrix-vector dimension mismatch");
    }

    std::vector<double> result(rows_, 0.0);
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            result[i] += (*this)(i, j) * vec[j];
        }
    }
    return result;
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

namespace vec_ops {

std::vector<double> sigmoid(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), sigmoid_func);
    return result;
}

std::vector<double> tanh(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(),
                   [](double x) { return std::tanh(x); });
    return result;
}

std::vector<double> concatenate(const std::vector<double>& v1,
                                const std::vector<double>& v2) {
    std::vector<double> result;
    result.reserve(v1.size() + v2.size());
    result.insert(result.end(), v1.begin(), v1.end());
    result.insert(result.end(), v2.begin(), v2.end());
    return result;
}

std::vector<double> element_mult(const std::vector<double>& v1,
                                 const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        result[i] = v1[i] * v2[i];
    }
    return result;
}

std::vector<double> add(const std::vector<double>& v1,
                       const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

std::vector<double> operator+(const std::vector<double>& v1,
                              const std::vector<double>& v2) {
    return add(v1, v2);
}

std::vector<double> operator*(double scalar, const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = scalar * vec[i];
    }
    return result;
}

std::vector<double> zeros(size_t size) {
    return std::vector<double>(size, 0.0);
}

std::vector<double> randn(size_t size) {
    std::vector<double> result(size);
    for (auto& val : result) {
        val = ::gru::randn();
    }
    return result;
}

} // namespace vec_ops

/* ============================================================================
 * GRUCell Implementation
 * ============================================================================ */

GRUCell::GRUCell(size_t input_dim, size_t hidden_dim)
    : input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      W_z_(hidden_dim, input_dim + hidden_dim, true),
      W_r_(hidden_dim, input_dim + hidden_dim, true),
      W_h_(hidden_dim, input_dim + hidden_dim, true),
      b_z_(vec_ops::randn(hidden_dim)),
      b_r_(vec_ops::randn(hidden_dim)),
      b_h_(vec_ops::randn(hidden_dim)) {}

std::vector<double> GRUCell::forward(const std::vector<double>& x,
                                     const std::vector<double>& h_prev) const {
    if (x.size() != input_dim_) {
        throw std::invalid_argument("Input dimension mismatch");
    }
    if (h_prev.size() != hidden_dim_) {
        throw std::invalid_argument("Hidden state dimension mismatch");
    }

    // Concatenate input and previous hidden state
    auto combined = vec_ops::concatenate(x, h_prev);

    // Calculate the update gate: z = sigmoid(W_z * combined + b_z)
    auto z_linear = vec_ops::add(W_z_ * combined, b_z_);
    auto z = vec_ops::sigmoid(z_linear);

    // Calculate the reset gate: r = sigmoid(W_r * combined + b_r)
    auto r_linear = vec_ops::add(W_r_ * combined, b_r_);
    auto r = vec_ops::sigmoid(r_linear);

    // Calculate the candidate hidden state: h_tilde = tanh(W_h * [x, r * h_prev] + b_h)
    auto r_h = vec_ops::element_mult(r, h_prev);
    auto combined_h = vec_ops::concatenate(x, r_h);
    auto h_tilde_linear = vec_ops::add(W_h_ * combined_h, b_h_);
    auto h_tilde = vec_ops::tanh(h_tilde_linear);

    // Update the hidden state: h = (1 - z) * h_prev + z * h_tilde
    std::vector<double> h(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; i++) {
        h[i] = (1.0 - z[i]) * h_prev[i] + z[i] * h_tilde[i];
    }

    return h;
}

/* ============================================================================
 * GRUArray Implementation
 * ============================================================================ */

GRUArray::GRUArray(size_t num_cells, size_t input_dim, size_t hidden_dim)
    : input_dim_(input_dim), hidden_dim_(hidden_dim) {
    cells_.reserve(num_cells);
    for (size_t i = 0; i < num_cells; i++) {
        cells_.emplace_back(input_dim, hidden_dim);
    }
}

std::vector<double> GRUArray::forward(const std::vector<double>& x,
                                     const std::vector<double>& h_prev) const {
    if (cells_.empty()) {
        return h_prev;
    }

    auto h = h_prev;
    for (const auto& cell : cells_) {
        h = cell.forward(x, h);
    }
    return h;
}

std::vector<double> GRUArray::process_batch(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<double>& initial_state) const {

    // Use provided initial state or create zeros
    auto h_prev = initial_state.empty() ? vec_ops::zeros(hidden_dim_) : initial_state;

    // Process each input in the batch
    for (const auto& x : inputs) {
        h_prev = forward(x, h_prev);
    }

    return h_prev;
}

} // namespace gru
