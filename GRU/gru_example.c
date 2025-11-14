/**
 * @file gru_example.c
 * @brief Example usage of GRU cell array in C
 */

#include "gru_array.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // Configuration
    const size_t num_cells = 10;
    const size_t input_dim = 32;
    const size_t hidden_dim = 16;
    const size_t batch_size = 5;

    printf("GRU Cell Array Example\n");
    printf("======================\n");
    printf("Number of cells: %zu\n", num_cells);
    printf("Input dimension: %zu\n", input_dim);
    printf("Hidden dimension: %zu\n", hidden_dim);
    printf("Batch size: %zu\n\n", batch_size);

    // Create an array of GRU cells
    GRUCell *gru_array = (GRUCell *)malloc(num_cells * sizeof(GRUCell));
    if (gru_array == NULL) {
        fprintf(stderr, "Error: Failed to allocate GRU array\n");
        return EXIT_FAILURE;
    }

    printf("Initializing GRU cells...\n");
    for (size_t i = 0; i < num_cells; i++) {
        gru_array[i] = gru_cell_create(input_dim, hidden_dim);
        printf("  Cell %zu initialized\n", i);
    }
    printf("\n");

    // Initialize the hidden state
    Vector h_prev = vector_create(hidden_dim);
    Vector h_next = vector_create(hidden_dim);
    vector_zeros(&h_prev);

    // Create input data (random for this example)
    printf("Processing batch of %zu inputs...\n", batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        // Create a random input vector
        Vector x = vector_create(input_dim);
        vector_randn(&x);

        // Process through the GRU array
        gru_array_forward(gru_array, num_cells, &x, &h_prev, &h_next);

        // Update hidden state for next iteration
        vector_copy(&h_next, &h_prev);

        // Print first few elements of hidden state for this batch item
        printf("  Batch %zu - Hidden state (first 5 elements): ", i);
        for (size_t j = 0; j < 5 && j < hidden_dim; j++) {
            printf("%.4f ", h_next.data[j]);
        }
        printf("...\n");

        // Free input vector
        vector_free(&x);
    }
    printf("\n");

    // Output the final hidden state
    printf("Final hidden state:\n");
    for (size_t i = 0; i < hidden_dim; i++) {
        printf("  h[%zu] = %.6f\n", i, h_next.data[i]);
    }
    printf("\n");

    // Cleanup
    printf("Cleaning up...\n");
    for (size_t i = 0; i < num_cells; i++) {
        gru_cell_free(&gru_array[i]);
    }
    free(gru_array);
    vector_free(&h_prev);
    vector_free(&h_next);

    printf("Done!\n");
    return EXIT_SUCCESS;
}
