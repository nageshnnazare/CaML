/**
 * @file nn_xor.c
 * @brief XOR logic gate implementation using a neural network.
 *
 * This file demonstrates how to train a simple neural network to approximate
 * the XOR logic gate functionality using finite differences for gradient
 * calculation and gradient descent for learning.
 */
#define NN_IMPLEMENTATION
#include "nn.h"

#include <time.h>

/**
 * @brief XOR Training data.
 *
 * Each row consists of two inputs (x1, x2) followed by the expected output (y).
 * Data pattern: {input1, input2, output}
 */
float train_data[] = {
    0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
};

/**
 * @brief Entry point: Trains and tests the XOR neural network.
 *
 * This function:
 * 1. Initializes the training and output matrices from `train_data`.
 * 2. Allocates a neural network with a 2-2-1 architecture (2 inputs, 2 hidden
 * neurons, 1 output).
 * 3. Trains the network for 20,000 iterations using finite differences.
 * 4. Prints the predicted XOR outcomes for all input combinations.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  srand(time(NULL));

  size_t stride = 3;
  size_t n = sizeof(train_data) / sizeof(train_data[0]) / stride;

  Matrix ti = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .data = train_data,
  };

  Matrix to = {
      .rows = n,
      .cols = 1,
      .stride = stride,
      .data = train_data + 2,
  };

  size_t arch[] = {2, 2, 1};

  NeuralNetwork nn = nn_alloc(arch, NN_ARRAY_LEN(arch));
  NeuralNetwork grad = nn_alloc(arch, NN_ARRAY_LEN(arch));
  nn_rand(nn, 0.0, 1.0);

  // Training loop
  for (size_t i = 0; i < 10 * 1000; i++) {
    float rate = 1e-1;
#ifdef XOR_FINITE_DIFF
    float eps = 1e-1;
    nn_finite_diff(nn, grad, eps, ti, to);
#else
    nn_backprop(nn, grad, ti, to);
#endif
    nn_learn(nn, grad, rate);
    // printf("Cost: %f\n", nn_cost(nn, ti, to));
  }

  // Testing/Verification loop
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
      MATRIX_AT(NN_INPUT(nn), 0, 1) = j;
      nn_forward(nn);
      printf("%zu ^ %zu = %f\n", i, j, MATRIX_AT(NN_OUTPUT(nn), 0, 0));
    }
  }

  nn_free(nn);
  nn_free(grad);

  return 0;
}