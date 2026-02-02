# CaML Core Library (`nn/`)

This directory contains the core matrix library and neural network implementations.

## Files

- [nn.h](nn.h): Modular single-header library for matrix operations.
- [nn_xor.c](nn_xor.c): Example XOR implementation using the modular library.
- [nn_test.c](nn_test.c): Test suite for matrix operations.

## Core Data Structures

### Matrix
The `Matrix` struct is the fundamental building block. It uses a flat array to represent a 2D matrix with a `stride` for efficient sub-matrix views.

```c
typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *data;
} Matrix;
```

## Key Functions

### Memory Management
- `matrix_alloc(rows, cols)`: Allocates memory for a new matrix.
- `matrix_free(m)`: Frees matrix memory.

### Math Operations
- `matrix_dot(res, a, b)`: Performs matrix multiplication.
- `matrix_sum(res, a, b)`: Adds two matrices into a third.
- `matrix_acc(res, a)`: Accumulates matrix `a` into `res`.
- `matrix_sigf(m)`: Applies sigmoid activation in-place.

### Utilities
- `matrix_print(m, name)`: Prints the matrix with ASCII box styling and auto-indentation.
- `matrix_rand(m, low, high)`: Fills with random values.
- `matrix_fill(m, val)`: Fills with a constant value.
- `matrix_row(m, row)`: Returns a row as a new matrix view.
- `matrix_copy(res, a)`: Copies contents between matrices.
