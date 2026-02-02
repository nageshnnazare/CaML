# MLC - Machine Learning from scratch in C

MLC is a lightweight, dependency-free neural network library written in C. It focuses on clarity and understanding the fundamentals of machine learning by implementing core concepts from scratch.

## Project Structure

- `nn/`: Core library containing matrix operations and neural network utilities.
  - `nn.h`: Single-header library with matrix implementations.
  - `nn.c`: Example usage/test for the `nn` library.
- `basics/`: Educational examples demonstrating simple ML concepts.
  - `ml_helloWorld.c`: Simple linear regression (y = wx + b).
  - `ml_withBoolean.c`: Single neuron training for OR/AND gates.
  - `ml_xor.c`: Multi-layer perception implementation for XOR gate.
- `Makefile`: Build instructions for all examples.
- `build/`: Directory for compiled binaries.

## How to Build

The project uses `clang` and a simple `Makefile`.

### Prerequisites

- `clang` (or any C compiler, but Makefile is set for clang)
- `make`

### Building Examples

To build all examples, run:
```bash
make
```

This will generate binaries in the `build/` directory:
- `build/ml_helloWorld`
- `build/ml_withBoolean`
- `build/ml_xor`
- `build/nn`

### Running an Example

Example for XOR gate:
```bash
./build/ml_xor
```

## Core Concepts

### Matrix Operations
The foundation of the library is a simple `Matrix` struct and associated operations (sum, dot product, allocation).

### Gradient Approximation
Instead of backpropagation (initially), the examples use **Finite Differences** to approximate gradients:
$$ f'(x) \approx \frac{f(x + \epsilon) - f(x)}{\epsilon} $$

This is an intuitive way to understand how weights and biases are updated to minimize the cost function.
