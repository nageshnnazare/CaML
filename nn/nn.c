#define NN_IMPLEMENTATION
#include "nn.h"

int main() {
  Matrix w1 = matrix_alloc(2, 2);
  Matrix b1 = matrix_alloc(1, 2);
  Matrix w2 = matrix_alloc(2, 1);
  Matrix b2 = matrix_alloc(1, 1);

  matrix_rand(w1, 0, 1);
  matrix_rand(b1, 0, 1);
  matrix_rand(w2, 0, 1);
  matrix_rand(b2, 0, 1);

  MAT_PRINT(w1);
  MAT_PRINT(b1);
  MAT_PRINT(w2);
  MAT_PRINT(b2);

  matrix_free(w1);
  matrix_free(b1);
  matrix_free(w2);
  matrix_free(b2);
}