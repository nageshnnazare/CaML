#define NN_IMPLEMENTATION
#include "nn.h"

int main() {
  {
    Matrix a = matrix_alloc(3, 3);
    matrix_fill(a, 1);
    Matrix b = matrix_alloc(3, 3);
    matrix_fill(b, 2);

    matrix_print(a);
    matrix_print(b);

    matrix_sum(a, b);
    matrix_print(a);

    matrix_free(a);
    matrix_free(b);
  }
  {
    Matrix a = matrix_alloc(1, 2);
    matrix_fill(a, 1);

    Matrix b = matrix_alloc(2, 3);
    matrix_fill(b, 1);

    Matrix c = matrix_alloc(1, 3);

    matrix_print(a);
    matrix_print(b);

    matrix_dot(c, a, b);
    matrix_print(c);
  }

  return 0;
}