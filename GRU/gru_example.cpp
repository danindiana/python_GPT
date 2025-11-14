/**
 * @file gru_example.cpp
 * @brief Example usage of GRU cell array in C++
 */

#include "gru_array.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // Configuration
    const size_t num_cells = 10;
    const size_t input_dim = 32;
    const size_t hidden_dim = 16;
    const size_t batch_size = 5;

    std::cout << "GRU Cell Array Example (C++)" << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "Number of cells: " << num_cells << std::endl;
    std::cout << "Input dimension: " << input_dim << std::endl;
    std::cout << "Hidden dimension: " << hidden_dim << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << std::endl;

    try {
        // Create a GRU array
        std::cout << "Initializing GRU array..." << std::endl;
        gru::GRUArray gru_array(num_cells, input_dim, hidden_dim);
        std::cout << "GRU array initialized with " << gru_array.num_cells() << " cells" << std::endl;
        std::cout << std::endl;

        // Create batch of random input data
        std::cout << "Generating random input batch..." << std::endl;
        std::vector<std::vector<double>> inputs;
        inputs.reserve(batch_size);
        for (size_t i = 0; i < batch_size; i++) {
            inputs.push_back(gru::vec_ops::randn(input_dim));
        }
        std::cout << "Generated " << inputs.size() << " input vectors" << std::endl;
        std::cout << std::endl;

        // Process batch using the convenient method
        std::cout << "Processing batch..." << std::endl;
        auto final_state = gru_array.process_batch(inputs);
        std::cout << "Batch processing complete" << std::endl;
        std::cout << std::endl;

        // Alternative: Process batch manually to show intermediate states
        std::cout << "Manual processing with intermediate states:" << std::endl;
        auto h_prev = gru::vec_ops::zeros(hidden_dim);
        for (size_t i = 0; i < batch_size; i++) {
            h_prev = gru_array.forward(inputs[i], h_prev);

            // Print first few elements of hidden state
            std::cout << "  Batch " << i << " - Hidden state (first 5 elements): ";
            for (size_t j = 0; j < 5 && j < hidden_dim; j++) {
                std::cout << std::fixed << std::setprecision(4) << h_prev[j] << " ";
            }
            std::cout << "..." << std::endl;
        }
        std::cout << std::endl;

        // Output the final hidden state
        std::cout << "Final hidden state:" << std::endl;
        for (size_t i = 0; i < hidden_dim; i++) {
            std::cout << "  h[" << i << "] = "
                     << std::fixed << std::setprecision(6) << final_state[i] << std::endl;
        }
        std::cout << std::endl;

        // Demonstrate single cell usage
        std::cout << "Demonstrating single GRU cell:" << std::endl;
        gru::GRUCell single_cell(input_dim, hidden_dim);
        auto single_input = gru::vec_ops::randn(input_dim);
        auto single_h_prev = gru::vec_ops::zeros(hidden_dim);
        auto single_h_next = single_cell.forward(single_input, single_h_prev);

        std::cout << "  Single cell output (first 5 elements): ";
        for (size_t i = 0; i < 5 && i < hidden_dim; i++) {
            std::cout << std::fixed << std::setprecision(4) << single_h_next[i] << " ";
        }
        std::cout << "..." << std::endl;
        std::cout << std::endl;

        std::cout << "Done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
