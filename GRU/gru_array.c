/**
 * @file gru_array.c
 * @brief C implementation of Gated Recurrent Unit (GRU) cell array
 */

#define _USE_MATH_DEFINES
#include "gru_array.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* Define M_PI if not available */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Random Number Generation
 * ============================================================================ */

static int random_initialized = 0;

/**
 * @brief Initialize random number generator (call once)
 */
static void init_random(void) {
    if (!random_initialized) {
        srand((unsigned int)time(NULL));
        random_initialized = 1;
    }
}

/**
 * @brief Generate random number from standard normal distribution (Box-Muller)
 * @return Random value from N(0,1)
 */
static double randn(void) {
    init_random();
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ============================================================================
 * Memory Management Functions
 * ============================================================================ */

Matrix matrix_create(size_t rows, size_t cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (double *)calloc(rows * cols, sizeof(double));
    if (mat.data == NULL) {
        fprintf(stderr, "Error: Failed to allocate matrix memory\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

Vector vector_create(size_t size) {
    Vector vec;
    vec.size = size;
    vec.data = (double *)calloc(size, sizeof(double));
    if (vec.data == NULL) {
        fprintf(stderr, "Error: Failed to allocate vector memory\n");
        exit(EXIT_FAILURE);
    }
    return vec;
}

void matrix_free(Matrix *mat) {
    if (mat != NULL && mat->data != NULL) {
        free(mat->data);
        mat->data = NULL;
        mat->rows = 0;
        mat->cols = 0;
    }
}

void vector_free(Vector *vec) {
    if (vec != NULL && vec->data != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->size = 0;
    }
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double tanh_activation(double x) {
    return tanh(x);
}

void vector_sigmoid(const Vector *vec, Vector *result) {
    for (size_t i = 0; i < vec->size; i++) {
        result->data[i] = sigmoid(vec->data[i]);
    }
}

void vector_tanh(const Vector *vec, Vector *result) {
    for (size_t i = 0; i < vec->size; i++) {
        result->data[i] = tanh_activation(vec->data[i]);
    }
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

void vector_concatenate(const Vector *v1, const Vector *v2, Vector *result) {
    memcpy(result->data, v1->data, v1->size * sizeof(double));
    memcpy(result->data + v1->size, v2->data, v2->size * sizeof(double));
}

void vector_element_mult(const Vector *v1, const Vector *v2, Vector *result) {
    for (size_t i = 0; i < v1->size; i++) {
        result->data[i] = v1->data[i] * v2->data[i];
    }
}

void vector_add(const Vector *v1, const Vector *v2, Vector *result) {
    for (size_t i = 0; i < v1->size; i++) {
        result->data[i] = v1->data[i] + v2->data[i];
    }
}

void vector_scale_add(double scalar, const Vector *v1, const Vector *v2, Vector *result) {
    for (size_t i = 0; i < v1->size; i++) {
        result->data[i] = scalar * v1->data[i] + v2->data[i];
    }
}

void vector_copy(const Vector *src, Vector *dst) {
    memcpy(dst->data, src->data, src->size * sizeof(double));
}

void vector_zeros(Vector *vec) {
    memset(vec->data, 0, vec->size * sizeof(double));
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

void matrix_vector_mult(const Matrix *mat, const Vector *vec, Vector *result) {
    for (size_t i = 0; i < mat->rows; i++) {
        result->data[i] = 0.0;
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[i] += mat->data[i * mat->cols + j] * vec->data[j];
        }
    }
}

void matrix_randn(Matrix *mat) {
    for (size_t i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = randn();
    }
}

void vector_randn(Vector *vec) {
    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] = randn();
    }
}

/* ============================================================================
 * GRU Cell Functions
 * ============================================================================ */

GRUCell gru_cell_create(size_t input_dim, size_t hidden_dim) {
    GRUCell cell;
    cell.input_dim = input_dim;
    cell.hidden_dim = hidden_dim;

    size_t combined_dim = input_dim + hidden_dim;

    // Allocate weight matrices (hidden_dim x combined_dim)
    cell.W_z = matrix_create(hidden_dim, combined_dim);
    cell.W_r = matrix_create(hidden_dim, combined_dim);
    cell.W_h = matrix_create(hidden_dim, combined_dim);

    // Allocate bias vectors
    cell.b_z = vector_create(hidden_dim);
    cell.b_r = vector_create(hidden_dim);
    cell.b_h = vector_create(hidden_dim);

    // Initialize with random values
    matrix_randn(&cell.W_z);
    matrix_randn(&cell.W_r);
    matrix_randn(&cell.W_h);
    vector_randn(&cell.b_z);
    vector_randn(&cell.b_r);
    vector_randn(&cell.b_h);

    return cell;
}

void gru_cell_free(GRUCell *cell) {
    if (cell != NULL) {
        matrix_free(&cell->W_z);
        matrix_free(&cell->W_r);
        matrix_free(&cell->W_h);
        vector_free(&cell->b_z);
        vector_free(&cell->b_r);
        vector_free(&cell->b_h);
    }
}

void gru_cell_forward(const GRUCell *cell, const Vector *x, const Vector *h_prev, Vector *h_next) {
    size_t hidden_dim = cell->hidden_dim;
    size_t combined_dim = cell->input_dim + hidden_dim;

    // Create temporary vectors
    Vector combined = vector_create(combined_dim);
    Vector z_linear = vector_create(hidden_dim);
    Vector z = vector_create(hidden_dim);
    Vector r_linear = vector_create(hidden_dim);
    Vector r = vector_create(hidden_dim);
    Vector r_h = vector_create(hidden_dim);
    Vector combined_h = vector_create(combined_dim);
    Vector h_tilde_linear = vector_create(hidden_dim);
    Vector h_tilde = vector_create(hidden_dim);
    Vector one_minus_z = vector_create(hidden_dim);
    Vector term1 = vector_create(hidden_dim);
    Vector term2 = vector_create(hidden_dim);

    // Concatenate input and previous hidden state
    vector_concatenate(x, h_prev, &combined);

    // Calculate the update gate: z = sigmoid(W_z * combined + b_z)
    matrix_vector_mult(&cell->W_z, &combined, &z_linear);
    vector_add(&z_linear, &cell->b_z, &z_linear);
    vector_sigmoid(&z_linear, &z);

    // Calculate the reset gate: r = sigmoid(W_r * combined + b_r)
    matrix_vector_mult(&cell->W_r, &combined, &r_linear);
    vector_add(&r_linear, &cell->b_r, &r_linear);
    vector_sigmoid(&r_linear, &r);

    // Calculate the candidate hidden state: h_tilde = tanh(W_h * [x, r * h_prev] + b_h)
    vector_element_mult(&r, h_prev, &r_h);
    vector_concatenate(x, &r_h, &combined_h);
    matrix_vector_mult(&cell->W_h, &combined_h, &h_tilde_linear);
    vector_add(&h_tilde_linear, &cell->b_h, &h_tilde_linear);
    vector_tanh(&h_tilde_linear, &h_tilde);

    // Update the hidden state: h = (1 - z) * h_prev + z * h_tilde
    for (size_t i = 0; i < hidden_dim; i++) {
        one_minus_z.data[i] = 1.0 - z.data[i];
    }
    vector_element_mult(&one_minus_z, h_prev, &term1);
    vector_element_mult(&z, &h_tilde, &term2);
    vector_add(&term1, &term2, h_next);

    // Free temporary vectors
    vector_free(&combined);
    vector_free(&z_linear);
    vector_free(&z);
    vector_free(&r_linear);
    vector_free(&r);
    vector_free(&r_h);
    vector_free(&combined_h);
    vector_free(&h_tilde_linear);
    vector_free(&h_tilde);
    vector_free(&one_minus_z);
    vector_free(&term1);
    vector_free(&term2);
}

/* ============================================================================
 * GRU Array Functions
 * ============================================================================ */

void gru_array_forward(const GRUCell *cells, size_t num_cells, const Vector *x,
                       const Vector *h_prev, Vector *h_next) {
    if (num_cells == 0) {
        vector_copy(h_prev, h_next);
        return;
    }

    // Temporary storage for intermediate hidden states
    Vector h_temp1 = vector_create(cells[0].hidden_dim);
    Vector h_temp2 = vector_create(cells[0].hidden_dim);
    Vector *current_h = &h_temp1;
    Vector *next_h = &h_temp2;

    // Process through first cell
    gru_cell_forward(&cells[0], x, h_prev, current_h);

    // Process through remaining cells
    for (size_t i = 1; i < num_cells; i++) {
        gru_cell_forward(&cells[i], x, current_h, next_h);
        // Swap pointers
        Vector *temp = current_h;
        current_h = next_h;
        next_h = temp;
    }

    // Copy final result
    vector_copy(current_h, h_next);

    // Free temporary vectors
    vector_free(&h_temp1);
    vector_free(&h_temp2);
}
