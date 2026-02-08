/**
 * @file nn_fa.c
 * @brief Neural Network based Multi-bit Adder (Full Adder concept).
 *
 * This example demonstrates how a neural network can learn to perform
 * binary addition of two multi-bit numbers. It generates all possible
 * combinations of inputs for a given bit-width, trains the network,
 * and سپس verifies the output.
 */

#define NN_IMPLEMENTATION
#include "../nn.h"

#include <time.h>

/**
 * @brief Number of bits for each input operand.
 * 2 bits means inputs range from 0-3, and output ranges from 0-6 (3 bits +
 * overflow).
 */
#define BITS 2

int main() {
  srand(time(NULL));

  size_t n = (1 << BITS); // Number of possible values for BITS bits
  size_t rows = n * n;    // Total combinations of (x, y)

  // Input NN_Matrix: (x_bits, y_bits)
  NN_Matrix ti = nn_mat_alloc(rows, BITS * 2);
  // Output NN_Matrix: (sum_bits, overflow_bit)
  NN_Matrix to = nn_mat_alloc(rows, BITS + 1);

  // Generate training data for all possible sums of x and y
  for (size_t i = 0; i < ti.rows; i++) {
    size_t x = i / n;
    size_t y = i % n;
    size_t z = x + y;
    for (size_t j = 0; j < BITS; j++) {
      // Decompose x and y into bits for input
      NN_MAT_AT(ti, i, j) = (x >> j) & 1;
      NN_MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
      // Decompose z (sum) into bits for target output
      NN_MAT_AT(to, i, j) = (z >> j) & 1;
    }
    // Set the overflow bit (carry out of the last bit)
    NN_MAT_AT(to, i, BITS) = z >= n;
  }

  // Define the network architecture: [Inputs, Hidden Layer, Outputs]
  size_t arch[] = {2 * BITS, 2 * BITS, BITS + 1};
  NN_NeuralNetwork nn = nn_alloc(arch, NN_ARRAY_LEN(arch));
  NN_NeuralNetwork grad = nn_alloc(arch, NN_ARRAY_LEN(arch));

  // Randomize initial weights
  nn_rand(nn, 0.0, 1.0);
  float rate = 1;

  // printf("Cost: %f\n", nn_cost(nn, ti, to));

  // Training loop: Adjust weights using backpropagation to minimize cost
  for (size_t i = 0; i < 20 * 1000; i++) {
    nn_backprop(nn, grad, ti, to);
    nn_learn(nn, grad, rate);
    // printf("%zu: Cost: %f\n", i, nn_cost(nn, ti, to));
  }

  // Verification: Test the trained network against all inputs
  size_t fail_count = 0;
  for (size_t x = 0; x < n; x++) {
    for (size_t y = 0; y < n; y++) {
      size_t z = x + y;
      // Load current x and y into the NN input layer
      for (size_t j = 0; j < BITS; j++) {
        NN_MAT_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
        NN_MAT_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
      }

      nn_forward(nn);

      // Check overflow bit
      if (NN_MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5) {
        if (z < n) {
          printf("%zu + %zu = (ov<>%zu)\n", x, y, z);
          fail_count++;
        }
      } else {
        // Reconstruct numeric sum from output bits
        size_t a = 0;
        for (size_t j = 0; j < BITS; j++) {
          size_t bit = NN_MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5;
          a |= bit << j;
        }
        // Verify reconstructed sum against actual sum
        if (a != z) {
          printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
          fail_count++;
        }
      }
    }
  }

  printf("Fail count: %zu\n", fail_count);

  // Cleanup
  nn_free(nn);
  nn_free(grad);
  nn_mat_free(ti);
  nn_mat_free(to);
  return 0;
}