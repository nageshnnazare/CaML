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
 * @var Matrix::stride The number of columns in the matrix.
 * @var Matrix::data Pointer to the flat-array of float values.
 */
typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *data;
} Matrix;

/**
 * @brief Accesses an element in the matrix at row i and column j.
 */
#define MATRIX_AT(m, i, j) ((m).data[(i) * (m).stride + (j)])

/**
 * @brief Prints the matrix contents to stdout.
 * @param m The matrix to print.
 * @param name The name of the matrix.
 */
#define MATRIX_PRINT(m) matrix_print(m, #m)

/**
 * @brief Generates a random float between 0.0 and 1.0.
 * @return float A random value.
 */
float randf(void);

/**
 * @brief Applies the sigmoid activation function to each element of the matrix.
 * @param x The value to apply the sigmoid function to.
 * @return float The result of the sigmoid function.
 */
float sigmoidf(float x);

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
 * @param name The name of the matrix.
 */
void matrix_print(Matrix m, const char *name);

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
 * @brief Returns a view of the specified row in the matrix.
 * @param m The matrix.
 * @param row The row index.
 * @return Matrix A view of the specified row.
 */
Matrix matrix_row(Matrix m, size_t row);

/**
 * @brief Returns a view of the specified sub-matrix of 'm'.
 * @param m The matrix.
 * @param row The starting row index.
 * @param col The starting column index.
 * @param rows The number of rows in the sub-matrix.
 * @param cols The number of columns in the sub-matrix.
 * @return Matrix A view of the specified sub-matrix.
 */
Matrix matrix_submatrix(Matrix m, size_t row, size_t col, size_t rows,
                        size_t cols);

/**
 * @brief Copies matrix 'a' into 'res' (res = a).
 * @param res The destination matrix.
 * @param a The source matrix.
 * @note Dimensions must match.
 */
void matrix_copy(Matrix res, Matrix a);

/**
 * @brief Applies the sigmoid activation function to each element of the matrix.
 * @param m The matrix to apply the sigmoid function to.
 */
void matrix_sigf(Matrix m);

/**
 * @brief Accumulates matrix 'a' into 'res' (res = res + a).
 * @param res The destination and first operand.
 * @param a The matrix to add.
 * @note Dimensions must match.
 */
void matrix_acc(Matrix res, Matrix a);

/**
 * @brief Adds matrix 'a' to 'b' and stores the result in 'res' (res = a + b).
 * @param res The destination matrix.
 * @param a The first operand.
 * @param b The second operand.
 * @note Dimensions must match.
 */
void matrix_sum(Matrix res, Matrix a, Matrix b);

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
#include <string.h>
#include <time.h>

float randf(void) { return (float)rand() / (float)RAND_MAX; }

float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

Matrix matrix_alloc(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.data = (float *)NN_MALLOC(rows * cols * sizeof(*m.data));
  NN_ASSERT(m.data != NULL);
  return m;
}

void matrix_free(Matrix m) { NN_FREE(m.data); }

void matrix_print(Matrix m, const char *name) {
  int prefix_len = printf("%s = ", name);
  for (size_t i = 0; i < m.rows; i++) {
    if (i > 0) {
      for (int k = 0; k < prefix_len; k++)
        printf(" ");
    }
    const char *start = "";
    const char *end = "";

    if (m.rows == 1) {
      start = "[ ";
      end = " ]";
    } else if (i == 0) {
      start = "┌ ";
      end = " ┐";
    } else if (i == m.rows - 1) {
      start = "└ ";
      end = " ┘";
    } else {
      start = "│ ";
      end = " │";
    }

    printf("%s", start);
    for (size_t j = 0; j < m.cols; j++) {
      printf("%10.6f ", MATRIX_AT(m, i, j));
    }
    printf("%s", end);

    if (i == m.rows - 1) {
      printf(" (%zu, %zu)", m.rows, m.cols);
    }
    printf("\n");
  }
}

void matrix_rand(Matrix m, float low, float high) {
  srand(time(NULL));
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MATRIX_AT(m, i, j) = low + (high - low) * randf();
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

Matrix matrix_row(Matrix m, size_t row) {
  return (Matrix){.rows = 1,
                  .cols = m.cols,
                  .stride = m.stride,
                  .data = &MATRIX_AT(m, row, 0)};
}

void matrix_copy(Matrix res, Matrix a) {
  NN_ASSERT(res.rows == a.rows);
  NN_ASSERT(res.cols == a.cols);
  memcpy(res.data, a.data, res.rows * res.cols * sizeof(*res.data));
}

void matrix_sigf(Matrix m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MATRIX_AT(m, i, j) = sigmoidf(MATRIX_AT(m, i, j));
    }
  }
}

void matrix_acc(Matrix res, Matrix a) {
  NN_ASSERT(res.rows == a.rows);
  NN_ASSERT(res.cols == a.cols);
  for (size_t i = 0; i < res.rows; i++) {
    for (size_t j = 0; j < res.cols; j++) {
      MATRIX_AT(res, i, j) += MATRIX_AT(a, i, j);
    }
  }
}

void matrix_sum(Matrix res, Matrix a, Matrix b) {
  NN_ASSERT(res.rows == a.rows);
  NN_ASSERT(res.cols == a.cols);
  NN_ASSERT(res.rows == b.rows);
  NN_ASSERT(res.cols == b.cols);
  for (size_t i = 0; i < res.rows; i++) {
    for (size_t j = 0; j < res.cols; j++) {
      MATRIX_AT(res, i, j) = MATRIX_AT(a, i, j) + MATRIX_AT(b, i, j);
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