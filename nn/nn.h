#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC(size) malloc(size)
#endif // NN_MALLOC

#ifndef NN_FREE
#include <stdlib.h>
#define NN_FREE(ptr) free(ptr)
#endif // NN_FREE

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT(condition) assert(condition)
#endif // NN_ASSERT

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;

#define MATRIX_AT(m, i, j) ((m).data[(i) * (m).cols + (j)])

float rand_float(void);

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_free(Matrix m);

void matrix_print(Matrix m);
void matrix_rand(Matrix m, float low, float high);
void matrix_fill(Matrix m, float val);

void matrix_sum(Matrix res, Matrix b);
void matrix_dot(Matrix res, Matrix a, Matrix b);

#endif // NN_H

#ifdef NN_IMPLEMENTATION
#include <math.h>
#include <stdio.h>
#include <time.h>

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

Matrix matrix_alloc(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.data = (float *)NN_MALLOC(rows * cols * sizeof(*m.data));
  NN_ASSERT(m.data != NULL);
  return m;
}

void matrix_free(Matrix m) { NN_FREE(m.data); }

void matrix_print(Matrix m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      printf("%f ", MATRIX_AT(m, i, j));
    }
    printf("\n");
  }
  printf("----------------------------\n");
}

void matrix_rand(Matrix m, float low, float high) {
  srand(time(NULL));
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MATRIX_AT(m, i, j) = low + (high - low) * rand_float();
    }
  }
}

void matrix_fill(Matrix m, float val) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MATRIX_AT(m, i, j) = val;
    }
  }
}

void matrix_sum(Matrix res, Matrix a) {
  NN_ASSERT(res.rows == a.rows);
  NN_ASSERT(res.cols == a.cols);
  for (size_t i = 0; i < res.rows; i++) {
    for (size_t j = 0; j < res.cols; j++) {
      MATRIX_AT(res, i, j) += MATRIX_AT(a, i, j);
    }
  }
}

void matrix_dot(Matrix res, Matrix a, Matrix b) {
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(res.rows == a.rows);
  NN_ASSERT(res.cols == b.cols);
  for (size_t i = 0; i < res.rows; i++) {
    for (size_t j = 0; j < res.cols; j++) {
      MATRIX_AT(res, i, j) = 0;
      for (size_t k = 0; k < n; k++) {
        MATRIX_AT(res, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
      }
    }
  }
}

#endif // NN_IMPLEMENTATION