# Makefile for GRU implementations in C and C++

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -std=c99 -Wall -O2
CXXFLAGS = -std=c++11 -Wall -O2
LDFLAGS = -lm

# Target executables
C_TARGET = gru_example_c
CPP_TARGET = gru_example_cpp

# Source files
C_SOURCES = gru_array.c gru_example.c
CPP_SOURCES = gru_array.cpp gru_example.cpp

# Object files
C_OBJECTS = gru_array.o gru_example.o
CPP_OBJECTS = gru_array_cpp.o gru_example_cpp.o

# Header files
C_HEADERS = gru_array.h
CPP_HEADERS = gru_array.hpp

# Default target: build everything
.PHONY: all
all: $(C_TARGET) $(CPP_TARGET)
	@echo "Build complete!"
	@echo "Run C version with: ./$(C_TARGET)"
	@echo "Run C++ version with: ./$(CPP_TARGET)"

# C target
$(C_TARGET): gru_array.o gru_example.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "C version compiled successfully!"

# C++ target
$(CPP_TARGET): gru_array_cpp.o gru_example_cpp.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "C++ version compiled successfully!"

# Compile C source files
gru_array.o: gru_array.c $(C_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

gru_example.o: gru_example.c $(C_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile C++ source files
gru_array_cpp.o: gru_array.cpp $(CPP_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

gru_example_cpp.o: gru_example.cpp $(CPP_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build only C version
.PHONY: c
c: $(C_TARGET)
	@echo "C version compiled successfully!"

# Build only C++ version
.PHONY: cpp
cpp: $(CPP_TARGET)
	@echo "C++ version compiled successfully!"

# Run C version
.PHONY: run-c
run-c: $(C_TARGET)
	@echo "Running C version..."
	@./$(C_TARGET)

# Run C++ version
.PHONY: run-cpp
run-cpp: $(CPP_TARGET)
	@echo "Running C++ version..."
	@./$(CPP_TARGET)

# Run both versions
.PHONY: run
run: run-c run-cpp

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(C_OBJECTS) $(CPP_OBJECTS) $(C_TARGET) $(CPP_TARGET)
	@echo "Cleaned build artifacts"

# Debug build with symbols
.PHONY: debug
debug: CFLAGS += -g -O0
debug: CXXFLAGS += -g -O0
debug: clean all
	@echo "Debug build complete (with symbols, no optimization)"

# Optimized build
.PHONY: optimized
optimized: CFLAGS += -O3 -march=native -flto
optimized: CXXFLAGS += -O3 -march=native -flto
optimized: clean all
	@echo "Optimized build complete (aggressive optimizations)"

# Help target
.PHONY: help
help:
	@echo "GRU Implementation Makefile"
	@echo "==========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Build both C and C++ versions (default)"
	@echo "  c          - Build only C version"
	@echo "  cpp        - Build only C++ version"
	@echo "  run-c      - Build and run C version"
	@echo "  run-cpp    - Build and run C++ version"
	@echo "  run        - Build and run both versions"
	@echo "  debug      - Build with debug symbols and no optimization"
	@echo "  optimized  - Build with aggressive optimizations"
	@echo "  clean      - Remove all build artifacts"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make           # Build everything"
	@echo "  make c         # Build only C version"
	@echo "  make run-cpp   # Build and run C++ version"
	@echo "  make optimized # Build optimized versions"
	@echo "  make clean     # Clean build artifacts"

# Declare phony targets
.PHONY: all c cpp run run-c run-cpp clean debug optimized help
