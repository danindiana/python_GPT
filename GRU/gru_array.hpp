/**
 * @file gru_array.hpp
 * @brief C++ implementation of Gated Recurrent Unit (GRU) cell array
 *
 * This header file defines the classes for implementing a GRU neural network
 * cell in modern C++ using RAII and standard library containers.
 */

#ifndef GRU_ARRAY_HPP
#define GRU_ARRAY_HPP

#include <vector>
#include <cstddef>
#include <memory>

namespace gru {

/**
 * @brief Matrix class using std::vector for storage (row-major order)
 */
class Matrix {
public:
    /**
     * @brief Construct a matrix with given dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(size_t rows, size_t cols);

    /**
     * @brief Construct a matrix and initialize with random values
     * @param rows Number of rows
     * @param cols Number of columns
     * @param random If true, initialize with random values from N(0,1)
     */
    Matrix(size_t rows, size_t cols, bool random);

    /**
     * @brief Get number of rows
     */
    size_t rows() const { return rows_; }

    /**
     * @brief Get number of columns
     */
    size_t cols() const { return cols_; }

    /**
     * @brief Access element at (i, j)
     */
    double& operator()(size_t i, size_t j);

    /**
     * @brief Access element at (i, j) (const version)
     */
    double operator()(size_t i, size_t j) const;

    /**
     * @brief Initialize matrix with random values from N(0,1)
     */
    void randomize();

    /**
     * @brief Matrix-vector multiplication
     * @param vec Input vector
     * @return Result vector
     */
    std::vector<double> operator*(const std::vector<double>& vec) const;

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
};

/**
 * @brief Vector operations namespace
 */
namespace vec_ops {
    /**
     * @brief Apply sigmoid element-wise to a vector
     * @param vec Input vector
     * @return Output vector
     */
    std::vector<double> sigmoid(const std::vector<double>& vec);

    /**
     * @brief Apply tanh element-wise to a vector
     * @param vec Input vector
     * @return Output vector
     */
    std::vector<double> tanh(const std::vector<double>& vec);

    /**
     * @brief Concatenate two vectors
     * @param v1 First vector
     * @param v2 Second vector
     * @return Concatenated vector
     */
    std::vector<double> concatenate(const std::vector<double>& v1,
                                    const std::vector<double>& v2);

    /**
     * @brief Element-wise multiplication of two vectors
     * @param v1 First vector
     * @param v2 Second vector
     * @return Result vector
     */
    std::vector<double> element_mult(const std::vector<double>& v1,
                                     const std::vector<double>& v2);

    /**
     * @brief Element-wise addition of two vectors
     * @param v1 First vector
     * @param v2 Second vector
     * @return Result vector
     */
    std::vector<double> add(const std::vector<double>& v1,
                           const std::vector<double>& v2);

    /**
     * @brief Vector addition: result = v1 + v2
     */
    std::vector<double> operator+(const std::vector<double>& v1,
                                  const std::vector<double>& v2);

    /**
     * @brief Scalar multiplication: result = scalar * vec
     */
    std::vector<double> operator*(double scalar, const std::vector<double>& vec);

    /**
     * @brief Create a vector of zeros
     * @param size Size of the vector
     * @return Vector of zeros
     */
    std::vector<double> zeros(size_t size);

    /**
     * @brief Create a vector with random values from N(0,1)
     * @param size Size of the vector
     * @return Vector with random values
     */
    std::vector<double> randn(size_t size);
}

/**
 * @brief GRU Cell class
 */
class GRUCell {
public:
    /**
     * @brief Construct a GRU cell with given dimensions
     * @param input_dim Dimension of input vector
     * @param hidden_dim Dimension of hidden state vector
     */
    GRUCell(size_t input_dim, size_t hidden_dim);

    /**
     * @brief Perform forward pass through GRU cell
     * @param x Input vector
     * @param h_prev Previous hidden state
     * @return Updated hidden state
     */
    std::vector<double> forward(const std::vector<double>& x,
                                const std::vector<double>& h_prev) const;

    /**
     * @brief Get input dimension
     */
    size_t input_dim() const { return input_dim_; }

    /**
     * @brief Get hidden dimension
     */
    size_t hidden_dim() const { return hidden_dim_; }

private:
    size_t input_dim_;
    size_t hidden_dim_;

    Matrix W_z_;  ///< Update gate weights
    Matrix W_r_;  ///< Reset gate weights
    Matrix W_h_;  ///< Candidate hidden state weights
    std::vector<double> b_z_;  ///< Update gate bias
    std::vector<double> b_r_;  ///< Reset gate bias
    std::vector<double> b_h_;  ///< Candidate hidden state bias
};

/**
 * @brief GRU Array class
 */
class GRUArray {
public:
    /**
     * @brief Construct a GRU array
     * @param num_cells Number of GRU cells in the array
     * @param input_dim Dimension of input vector
     * @param hidden_dim Dimension of hidden state vector
     */
    GRUArray(size_t num_cells, size_t input_dim, size_t hidden_dim);

    /**
     * @brief Process input through all GRU cells in sequence
     * @param x Input vector
     * @param h_prev Previous hidden state
     * @return Final hidden state after processing through all cells
     */
    std::vector<double> forward(const std::vector<double>& x,
                               const std::vector<double>& h_prev) const;

    /**
     * @brief Process a batch of inputs
     * @param inputs Batch of input vectors
     * @param initial_state Initial hidden state (default: zeros)
     * @return Final hidden state after processing all inputs
     */
    std::vector<double> process_batch(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<double>& initial_state = std::vector<double>()) const;

    /**
     * @brief Get number of cells
     */
    size_t num_cells() const { return cells_.size(); }

    /**
     * @brief Get input dimension
     */
    size_t input_dim() const { return input_dim_; }

    /**
     * @brief Get hidden dimension
     */
    size_t hidden_dim() const { return hidden_dim_; }

private:
    size_t input_dim_;
    size_t hidden_dim_;
    std::vector<GRUCell> cells_;
};

} // namespace gru

#endif // GRU_ARRAY_HPP
