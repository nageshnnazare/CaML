/**
 * @file nn_test.c
 * @brief Example usage and test suite for the nn.h matrix library.
 */

#define NN_IMPLEMENTATION
#include "../nn.h"

int main() {
  printf("--- NN_Matrix Addition Test ---\n");
  {
    NN_Matrix a = nn_mat_alloc(3, 3);
    nn_mat_fill(a, 1);
    NN_Matrix b = nn_mat_alloc(3, 3);
    nn_mat_fill(b, 2);

    nn_mat_print(a, "a", 0);
    nn_mat_print(b, "b", 0);

    nn_mat_acc(a, b);
    nn_mat_print(a, "a + b", 0);

    nn_mat_free(a);
    nn_mat_free(b);
  }
  printf("--- NN_Matrix Multiplication Test ---\n");
  {
    NN_Matrix a = nn_mat_alloc(1, 2);
    nn_mat_fill(a, 1);

    NN_Matrix b = nn_mat_alloc(2, 3);
    nn_mat_fill(b, 1);

    NN_Matrix c = nn_mat_alloc(1, 3);

    nn_mat_print(a, "a", 0);
    nn_mat_print(b, "b", 0);

    nn_mat_dot(c, a, b);
    nn_mat_print(c, "c", 0);

    nn_mat_free(a);
    nn_mat_free(b);
    nn_mat_free(c);
  }

  return 0;
}