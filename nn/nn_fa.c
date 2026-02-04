#define NN_IMPLEMENTATION
#include "nn.h"

#include <time.h>

#define BITS 2

int main() {
  srand(time(NULL));

  size_t n = (1 << BITS);
  size_t rows = n * n;
  Matrix ti = matrix_alloc(rows, BITS * 2);
  Matrix to = matrix_alloc(rows, BITS + 1);

  for (size_t i = 0; i < ti.rows; i++) {
    size_t x = i / n;
    size_t y = i % n;
    size_t z = x + y;
    for (size_t j = 0; j < BITS; j++) {
      MATRIX_AT(ti, i, j) = (x >> j) & 1;
      MATRIX_AT(ti, i, j + BITS) = (y >> j) & 1;
      MATRIX_AT(to, i, j) = (z >> j) & 1;
    }
    MATRIX_AT(to, i, BITS) = z >= n;
  }

  size_t arch[] = {2 * BITS, 2 * BITS, BITS + 1};
  NeuralNetwork nn = nn_alloc(arch, NN_ARRAY_LEN(arch));
  NeuralNetwork grad = nn_alloc(arch, NN_ARRAY_LEN(arch));
  nn_rand(nn, 0.0, 1.0);
  float rate = 1;

  // printf("Cost: %f\n", nn_cost(nn, ti, to));

  for (size_t i = 0; i < 20 * 1000; i++) {
    nn_backprop(nn, grad, ti, to);
    nn_learn(nn, grad, rate);
    //printf("%zu: Cost: %f\n", i, nn_cost(nn, ti, to));
  }

  size_t fail_count = 0;
  for (size_t x = 0; x < n; x++) {
    for (size_t y = 0; y < n; y++) {
      size_t z = x + y;
      for (size_t j = 0; j < BITS; j++) {
        MATRIX_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
        MATRIX_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
      }
      nn_forward(nn);
      if (MATRIX_AT(NN_OUTPUT(nn), 0, BITS) > 0.5) {
        if (z < n) {
          printf("%zu + %zu = (ov<>%zu)\n", x, y, z);
          fail_count++;
        }
      } else {
        size_t a = 0;
        for (size_t j = 0; j < BITS; j++) {
          size_t bit = MATRIX_AT(NN_OUTPUT(nn), 0, j) > 0.5;
          a |= bit << j;
        }
        if (a != z) {
          printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
          fail_count++;
        }
      }
    }
  }

  printf("Fail count: %zu\n", fail_count);

  nn_free(nn);
  nn_free(grad);
  matrix_free(ti);
  matrix_free(to);
  return 0;
}