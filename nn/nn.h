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

/**
 * @struct Matrix
 * @brief Represents a mathematical matrix of floats.
 *
 * @var Matrix::rows The number of rows in the matrix.
 * @var Matrix::cols The number of columns in the matrix.
 * @var Matrix::data Pointer to the flat-array of float values.
 */
typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;

/**
 * @brief Accesses an element in the matrix at row i and column j.
 */
#define MATRIX_AT(m, i, j) ((m).data[(i) * (m).cols + (j)])

/**
 * @brief Generates a random float between 0.0 and 1.0.
 * @return float A random value.
 */
float rand_float(void);

/**
 * @brief Allocates memory for a matrix of given size.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Matrix The allocated matrix.
 */
Matrix matrix_alloc(size_t rows, size_t cols);

/**
 * @brief Frees the memory allocated for a matrix.
 * @param m The matrix to free.
 */
void matrix_free(Matrix m);

/**
 * @brief Prints the matrix contents to stdout.
 * @param m The matrix to print.
 */
void matrix_print(Matrix m);

/**
 * @brief Fills the matrix with random values between low and high.
 * @param m The matrix to fill.
 * @param low The lower bound.
 * @param high The upper bound.
 */
void matrix_rand(Matrix m, float low, float high);

/**
 * @brief Fills the entire matrix with a single value.
 * @param m The matrix to fill.
 * @param val The value to set.
 */
void matrix_fill(Matrix m, float val);

/**
 * @brief Adds matrix 'a' to 'res' (res = res + a).
 * @param res The destination and first operand.
 * @param a The matrix to add.
 * @note Dimensions must match.
 */
void matrix_sum(Matrix res, Matrix a);

/**
 * @brief Performs matrix multiplication: res = a * b.
 * @param res The destination matrix.
 * @param a The left operand.
 * @param b The right operand.
 * @note Dimensions must be compatible: a.cols == b.rows.
 */
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