# GRU Implementation in C and C++

This directory contains C and C++ implementations of Gated Recurrent Unit (GRU) neural network cells, along with the original Python implementation.

## Overview

A Gated Recurrent Unit (GRU) is a type of recurrent neural network architecture that uses gating mechanisms to control information flow. This implementation includes:

- **Python version**: `gru_array.py` - NumPy-based implementation
- **C version**: `gru_array.c` and `gru_array.h` - Pure C implementation
- **C++ version**: `gru_array.cpp` and `gru_array.hpp` - Modern C++ implementation with RAII

## Files

### Python Implementation
- `gru_array.py` - Original Python implementation using NumPy
- `gru_array_readme.txt` - Explanation of the Python implementation

### C Implementation
- `gru_array.h` - C header file with structures and function declarations
- `gru_array.c` - C implementation with all GRU logic
- `gru_example.c` - Example program demonstrating C usage

### C++ Implementation
- `gru_array.hpp` - C++ header file with class definitions
- `gru_array.cpp` - C++ implementation using modern C++ features
- `gru_example.cpp` - Example program demonstrating C++ usage

### Build Files
- `Makefile` - Build configuration for compiling C and C++ versions

## Features

### C Implementation Features
- Pure C implementation (C99 standard)
- Manual memory management with create/free functions
- Matrix and vector operations
- Activation functions (sigmoid, tanh)
- GRU cell forward pass
- Array of GRU cells for multi-layer processing

### C++ Implementation Features
- Modern C++ (C++11 and later)
- RAII for automatic memory management
- STL containers (`std::vector`)
- Exception handling
- Namespace organization
- Operator overloading for intuitive syntax
- Batch processing utility

## Mathematical Background

A GRU cell computes the following operations:

1. **Update gate**: `z = σ(W_z · [x, h_prev] + b_z)`
2. **Reset gate**: `r = σ(W_r · [x, h_prev] + b_r)`
3. **Candidate hidden state**: `h_tilde = tanh(W_h · [x, r ⊙ h_prev] + b_h)`
4. **New hidden state**: `h = (1 - z) ⊙ h_prev + z ⊙ h_tilde`

Where:
- `σ` is the sigmoid activation function
- `⊙` denotes element-wise multiplication
- `[x, h_prev]` denotes concatenation of input and previous hidden state
- `W_*` are weight matrices
- `b_*` are bias vectors

## Compilation

### Prerequisites

**For C:**
- GCC or Clang compiler with C99 support
- Math library (`-lm`)

**For C++:**
- G++ or Clang++ compiler with C++11 support or later
- Math library (`-lm`)

### Using Makefile

Build all targets:
```bash
make all
```

Build only C version:
```bash
make c
```

Build only C++ version:
```bash
make cpp
```

Clean build artifacts:
```bash
make clean
```

### Manual Compilation

**C version:**
```bash
gcc -std=c99 -O2 -o gru_example_c gru_array.c gru_example.c -lm
```

**C++ version:**
```bash
g++ -std=c++11 -O2 -o gru_example_cpp gru_array.cpp gru_example.cpp -lm
```

### Compilation Flags Explained

- `-std=c99` or `-std=c++11`: Use C99 or C++11 standard
- `-O2`: Enable optimization level 2
- `-lm`: Link with math library (for `exp`, `tanh`, etc.)
- `-Wall`: Enable all warnings (recommended for development)
- `-g`: Include debugging symbols (for debugging)

## Usage

### Running the Examples

After compilation, run the example programs:

**C version:**
```bash
./gru_example_c
```

**C++ version:**
```bash
./gru_example_cpp
```

### C API Example

```c
#include "gru_array.h"

int main(void) {
    // Configuration
    size_t num_cells = 10;
    size_t input_dim = 32;
    size_t hidden_dim = 16;

    // Create GRU cells
    GRUCell *cells = malloc(num_cells * sizeof(GRUCell));
    for (size_t i = 0; i < num_cells; i++) {
        cells[i] = gru_cell_create(input_dim, hidden_dim);
    }

    // Initialize hidden state
    Vector h_prev = vector_create(hidden_dim);
    Vector h_next = vector_create(hidden_dim);
    vector_zeros(&h_prev);

    // Create input
    Vector x = vector_create(input_dim);
    vector_randn(&x);

    // Forward pass through GRU array
    gru_array_forward(cells, num_cells, &x, &h_prev, &h_next);

    // Cleanup
    for (size_t i = 0; i < num_cells; i++) {
        gru_cell_free(&cells[i]);
    }
    free(cells);
    vector_free(&x);
    vector_free(&h_prev);
    vector_free(&h_next);

    return 0;
}
```

### C++ API Example

```cpp
#include "gru_array.hpp"
#include <iostream>

int main() {
    // Configuration
    size_t num_cells = 10;
    size_t input_dim = 32;
    size_t hidden_dim = 16;

    // Create GRU array (memory managed automatically)
    gru::GRUArray gru_array(num_cells, input_dim, hidden_dim);

    // Create input batch
    std::vector<std::vector<double>> inputs;
    for (size_t i = 0; i < 5; i++) {
        inputs.push_back(gru::vec_ops::randn(input_dim));
    }

    // Process batch (initial state defaults to zeros)
    auto final_state = gru_array.process_batch(inputs);

    // Print results
    for (size_t i = 0; i < final_state.size(); i++) {
        std::cout << "h[" << i << "] = " << final_state[i] << std::endl;
    }

    return 0;
    // Memory automatically freed when objects go out of scope
}
```

## Performance Considerations

### C Implementation
- **Pros**:
  - Minimal overhead
  - Direct memory control
  - Suitable for embedded systems
- **Cons**:
  - Manual memory management required
  - More verbose code
  - No built-in error handling (returns need to be checked)

### C++ Implementation
- **Pros**:
  - Automatic memory management (RAII)
  - Exception handling
  - More intuitive API
  - Better type safety
- **Cons**:
  - Slightly higher overhead due to abstractions
  - Larger binary size

### Optimization Tips

1. **Compiler optimizations**: Use `-O2` or `-O3` flags
2. **Link-time optimization**: Add `-flto` flag
3. **Architecture-specific**: Add `-march=native` for CPU-specific optimizations
4. **Profiling**: Use tools like `gprof` or `valgrind` to identify bottlenecks

Example optimized compilation:
```bash
# C version with aggressive optimizations
gcc -std=c99 -O3 -march=native -flto -o gru_example_c gru_array.c gru_example.c -lm

# C++ version with aggressive optimizations
g++ -std=c++11 -O3 -march=native -flto -o gru_example_cpp gru_array.cpp gru_example.cpp -lm
```

## Memory Requirements

For a GRU cell with input dimension `I` and hidden dimension `H`:

- **Weight matrices**: 3 × (I + H) × H × sizeof(double)
- **Bias vectors**: 3 × H × sizeof(double)
- **Temporary storage during forward pass**: ~10 × H × sizeof(double)

Example for I=32, H=16 with 10 cells:
- Total memory ≈ 10 × (3 × 48 × 16 + 3 × 16 + 10 × 16) × 8 bytes ≈ 186 KB

## Testing

To verify correctness, you can compare outputs between Python, C, and C++ implementations:

1. Use the same random seed in all implementations
2. Process the same input data
3. Compare final hidden states (should be identical within floating-point precision)

## Limitations and Future Work

Current limitations:
- No backward pass (training) implementation
- No GPU acceleration
- Fixed double precision (could add float support)
- Random initialization only (no weight loading/saving)

Potential enhancements:
- Add backpropagation for training
- Implement weight serialization
- Add BLAS/LAPACK integration for faster matrix operations
- SIMD optimizations for vector operations
- GPU implementation using CUDA or OpenCL

## License

This implementation is provided as-is for educational and research purposes.

## References

1. Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
2. Chung, J., et al. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"

## Troubleshooting

### Common Compilation Issues

**Error: `undefined reference to 'exp'` or `'tanh'`**
- Solution: Add `-lm` flag to link with math library

**Error: `'for' loop initial declarations are only allowed in C99 mode`**
- Solution: Add `-std=c99` flag for C compilation

**Error: `'vector' does not name a type`**
- Solution: Ensure you're compiling C++ files with `g++`, not `gcc`

**Segmentation fault**
- Check array bounds
- Verify all vectors/matrices are properly allocated
- Ensure dimensions match between operations

## Contact

For issues or questions, please refer to the main repository documentation.
