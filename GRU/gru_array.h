/**
 * @file gru_array.h
 * @brief C implementation of Gated Recurrent Unit (GRU) cell array
 *
 * This header file defines the structures and functions for implementing
 * a GRU neural network cell in C.
 */

#ifndef GRU_ARRAY_H
#define GRU_ARRAY_H

#include <stddef.h>

/**
 * @brief Matrix structure for storing 2D arrays
 */
typedef struct {
    double *data;    /**< Pointer to matrix data (row-major order) */
    size_t rows;     /**< Number of rows */
    size_t cols;     /**< Number of columns */
} Matrix;

/**
 * @brief Vector structure for storing 1D arrays
 */
typedef struct {
    double *data;    /**< Pointer to vector data */
    size_t size;     /**< Size of the vector */
} Vector;

/**
 * @brief GRU Cell structure containing weights and biases
 */
typedef struct {
    Matrix W_z;      /**< Update gate weights */
    Matrix W_r;      /**< Reset gate weights */
    Matrix W_h;      /**< Candidate hidden state weights */
    Vector b_z;      /**< Update gate bias */
    Vector b_r;      /**< Reset gate bias */
    Vector b_h;      /**< Candidate hidden state bias */
    size_t input_dim;  /**< Input dimension */
    size_t hidden_dim; /**< Hidden state dimension */
} GRUCell;

/* ============================================================================
 * Memory Management Functions
 * ============================================================================ */

/**
 * @brief Create a new matrix with given dimensions
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Allocated matrix structure
 */
Matrix matrix_create(size_t rows, size_t cols);

/**
 * @brief Create a new vector with given size
 * @param size Size of the vector
 * @return Allocated vector structure
 */
Vector vector_create(size_t size);

/**
 * @brief Free matrix memory
 * @param mat Pointer to matrix to free
 */
void matrix_free(Matrix *mat);

/**
 * @brief Free vector memory
 * @param vec Pointer to vector to free
 */
void vector_free(Vector *vec);

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return sigmoid(x)
 */
double sigmoid(double x);

/**
 * @brief Hyperbolic tangent activation function
 * @param x Input value
 * @return tanh(x)
 */
double tanh_activation(double x);

/**
 * @brief Apply sigmoid element-wise to a vector
 * @param vec Input vector
 * @param result Output vector (must be pre-allocated)
 */
void vector_sigmoid(const Vector *vec, Vector *result);

/**
 * @brief Apply tanh element-wise to a vector
 * @param vec Input vector
 * @param result Output vector (must be pre-allocated)
 */
void vector_tanh(const Vector *vec, Vector *result);

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

/**
 * @brief Concatenate two vectors
 * @param v1 First vector
 * @param v2 Second vector
 * @param result Output vector (must be pre-allocated with size v1.size + v2.size)
 */
void vector_concatenate(const Vector *v1, const Vector *v2, Vector *result);

/**
 * @brief Element-wise multiplication of two vectors
 * @param v1 First vector
 * @param v2 Second vector
 * @param result Output vector (must be pre-allocated)
 */
void vector_element_mult(const Vector *v1, const Vector *v2, Vector *result);

/**
 * @brief Element-wise addition of two vectors
 * @param v1 First vector
 * @param v2 Second vector
 * @param result Output vector (must be pre-allocated)
 */
void vector_add(const Vector *v1, const Vector *v2, Vector *result);

/**
 * @brief Scalar-vector multiplication followed by addition: result = scalar * v1 + v2
 * @param scalar Scalar value
 * @param v1 First vector
 * @param v2 Second vector
 * @param result Output vector (must be pre-allocated)
 */
void vector_scale_add(double scalar, const Vector *v1, const Vector *v2, Vector *result);

/**
 * @brief Copy vector data
 * @param src Source vector
 * @param dst Destination vector (must be pre-allocated)
 */
void vector_copy(const Vector *src, Vector *dst);

/**
 * @brief Set all elements of a vector to zero
 * @param vec Vector to zero out
 */
void vector_zeros(Vector *vec);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

/**
 * @brief Matrix-vector multiplication: result = mat * vec
 * @param mat Input matrix
 * @param vec Input vector
 * @param result Output vector (must be pre-allocated)
 */
void matrix_vector_mult(const Matrix *mat, const Vector *vec, Vector *result);

/**
 * @brief Initialize matrix with random values from standard normal distribution
 * @param mat Matrix to initialize
 */
void matrix_randn(Matrix *mat);

/**
 * @brief Initialize vector with random values from standard normal distribution
 * @param vec Vector to initialize
 */
void vector_randn(Vector *vec);

/* ============================================================================
 * GRU Cell Functions
 * ============================================================================ */

/**
 * @brief Create and initialize a GRU cell
 * @param input_dim Dimension of input vector
 * @param hidden_dim Dimension of hidden state vector
 * @return Initialized GRU cell
 */
GRUCell gru_cell_create(size_t input_dim, size_t hidden_dim);

/**
 * @brief Free GRU cell memory
 * @param cell Pointer to GRU cell to free
 */
void gru_cell_free(GRUCell *cell);

/**
 * @brief Perform forward pass through GRU cell
 * @param cell GRU cell
 * @param x Input vector
 * @param h_prev Previous hidden state
 * @param h_next Output hidden state (must be pre-allocated)
 */
void gru_cell_forward(const GRUCell *cell, const Vector *x, const Vector *h_prev, Vector *h_next);

/* ============================================================================
 * GRU Array Functions
 * ============================================================================ */

/**
 * @brief Process input through an array of GRU cells
 * @param cells Array of GRU cells
 * @param num_cells Number of cells in the array
 * @param x Input vector
 * @param h_prev Previous hidden state
 * @param h_next Output hidden state (must be pre-allocated)
 */
void gru_array_forward(const GRUCell *cells, size_t num_cells, const Vector *x,
                       const Vector *h_prev, Vector *h_next);

#endif /* GRU_ARRAY_H */
