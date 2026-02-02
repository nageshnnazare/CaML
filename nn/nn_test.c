/**
 * @file nn_test.c
 * @brief Example usage and test suite for the nn.h matrix library.
 */

#define NN_IMPLEMENTATION
#include "nn.h"

int main() {
  printf("--- Matrix Addition Test ---\n");
  {
    Matrix a = matrix_alloc(3, 3);
    matrix_fill(a, 1);
    Matrix b = matrix_alloc(3, 3);
    matrix_fill(b, 2);

    matrix_print(a, "a");
    matrix_print(b, "b");

    matrix_sum(a, b);
    matrix_print(a, "a + b");

    matrix_free(a);
    matrix_free(b);
  }
  printf("--- Matrix Multiplication Test ---\n");
  {
    Matrix a = matrix_alloc(1, 2);
    matrix_fill(a, 1);

    Matrix b = matrix_alloc(2, 3);
    matrix_fill(b, 1);

    Matrix c = matrix_alloc(1, 3);

    matrix_print(a, "a");
    matrix_print(b, "b");

    matrix_dot(c, a, b);
    matrix_print(c, "c");

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
  }

  return 0;
}