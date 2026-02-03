#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>

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
 * @brief Returns the number of elements in an array.
 * @param arr Array.
 * @return Number of elements.
 */
#define NN_ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * @brief Allocates memory for a matrix.
 * @param size The size of the memory to allocate.
 * @return void* Pointer to the allocated memory.
 */
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC(size) malloc(size)
#endif // NN_MALLOC

/**
 * @brief Frees the memory allocated for a matrix.
 * @param ptr Pointer to the memory to free.
 */
#ifndef NN_FREE
#include <stdlib.h>
#define NN_FREE(ptr) free(ptr)
#endif // NN_FREE

/**
 * @brief Asserts a condition.
 * @param condition The condition to assert.
 */
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
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)

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
 * @param padding The padding to use for the neural network.
 */
void matrix_print(Matrix m, const char *name, int padding);

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

/**
 * @brief Structure to represent a neural network.
 * @var NeuralNetwork::count The number of layers in the neural network.
 * @var NeuralNetwork::ws The weights of the neural network.
 * @var NeuralNetwork::bs The biases of the neural network.
 * @var NeuralNetwork::as The activations of the neural network.
 */
typedef struct {
  size_t count;
  Matrix *ws;
  Matrix *bs;
  Matrix *as;
} NeuralNetwork;

/**
 * @brief Prints the neural network.
 * @param nn The neural network to print.
 */
#define NN_PRINT(nn) nn_print(nn, #nn)

/**
 * @brief Returns the input matrix of the neural network.
 * @param nn The neural network.
 * @return The input matrix.
 */
#define NN_INPUT(nn) ((nn).as[0])

/**
 * @brief Returns the output matrix of the neural network.
 * @param nn The neural network.
 * @return The output matrix.
 */
#define NN_OUTPUT(nn) ((nn).as[(nn).count])

/**
 * @brief Allocates memory for a neural network.
 * @param arch Array of layer sizes.
 * @param arch_count Number of layers.
 * @return NeuralNetwork.
 */
NeuralNetwork nn_alloc(size_t *arch, size_t arch_count);

/**
 * @brief Frees the memory allocated for a neural network.
 * @param nn The neural network to free.
 */
void nn_free(NeuralNetwork nn);

/**
 * @brief Prints the neural network.
 * @param nn The neural network to print.
 * @param name The name of the neural network.
 */
void nn_print(NeuralNetwork nn, const char *name);

/**
 * @brief Initializes the neural network with random weights and biases.
 * @param nn The neural network to initialize.
 * @param low The lower bound of the random values.
 * @param high The upper bound of the random values.
 */
void nn_rand(NeuralNetwork nn, float low, float high);

/**
 * @brief Forward pass of the neural network.
 * @param nn The neural network.
 */
void nn_forward(NeuralNetwork nn);

/**
 * @brief Calculates the cost of the neural network.
 * @param nn The neural network.
 * @param ti The input matrix.
 * @param to The output matrix.
 * @return The cost of the neural network.
 */
float nn_cost(NeuralNetwork nn, Matrix ti, Matrix to);

/**
 * @brief Calculates the gradient of the neural network.
 * @param nn The neural network.
 * @param grad The gradient of the neural network.
 * @param eps The epsilon value.
 * @param ti The input matrix.
 * @param to The output matrix.
 */
void nn_finite_diff(NeuralNetwork nn, NeuralNetwork grad, float eps, Matrix ti,
                    Matrix to);

/**
 * @brief Updates the neural network with the gradient.
 * @param nn The neural network.
 * @param grad The gradient of the neural network.
 * @param rate The learning rate.
 */
void nn_learn(NeuralNetwork nn, NeuralNetwork grad, float rate);

#endif // NN_H

#ifdef NN_IMPLEMENTATION
#include <math.h>
#include <stdio.h>
#include <string.h>

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

void matrix_print(Matrix m, const char *name, int padding) {
  printf("%*s", padding, "");
  int prefix_len = printf("%s = ", name);
  for (size_t i = 0; i < m.rows; i++) {
    if (i > 0) {
      printf("%*s", padding, "");
      printf("%*s", prefix_len, "");
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

NeuralNetwork nn_alloc(size_t *arch, size_t arch_count) {
  NeuralNetwork nn;
  NN_ASSERT(arch_count > 0);
  nn.count = arch_count - 1;
  nn.ws = (Matrix *)NN_MALLOC(nn.count * sizeof(*nn.ws));
  NN_ASSERT(nn.ws != NULL);
  nn.bs = (Matrix *)NN_MALLOC(nn.count * sizeof(*nn.bs));
  NN_ASSERT(nn.bs != NULL);
  nn.as = (Matrix *)NN_MALLOC((nn.count + 1) * sizeof(*nn.as));
  NN_ASSERT(nn.as != NULL);

  for (size_t i = 0; i < nn.count; i++) {
    nn.ws[i] = matrix_alloc(arch[i], arch[i + 1]);
    nn.bs[i] = matrix_alloc(1, arch[i + 1]);
    nn.as[i] = matrix_alloc(1, arch[i]);
  }
  nn.as[nn.count] = matrix_alloc(1, arch[nn.count]);
  return nn;
}

void nn_free(NeuralNetwork nn) {
  for (size_t i = 0; i < nn.count; i++) {
    matrix_free(nn.ws[i]);
    matrix_free(nn.bs[i]);
    matrix_free(nn.as[i]);
  }
  matrix_free(nn.as[nn.count]);
  NN_FREE(nn.ws);
  NN_FREE(nn.bs);
  NN_FREE(nn.as);
}

void nn_print(NeuralNetwork nn, const char *name) {
  char buf[256];
  int padding = 4;
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; i++) {
    printf("%*s", padding, "");
    printf("%s %zu:\n", "#Layer", i);
    snprintf(buf, sizeof(buf), "ws[%zu]", i);
    matrix_print(nn.ws[i], buf, padding + 2);
    snprintf(buf, sizeof(buf), "bs[%zu]", i);
    matrix_print(nn.bs[i], buf, padding + 2);
    snprintf(buf, sizeof(buf), "as[%zu]", i);
    matrix_print(nn.as[i], buf, padding + 2);
    printf("\n");
  }
  printf("%*s", padding, "");
  printf("%s:\n", "#Output");
  snprintf(buf, sizeof(buf), "as[%zu]", nn.count);
  matrix_print(nn.as[nn.count], buf, padding + 2);
  printf("]\n");
}

void nn_rand(NeuralNetwork nn, float low, float high) {
  for (size_t i = 0; i < nn.count; i++) {
    matrix_rand(nn.ws[i], low, high);
    matrix_rand(nn.bs[i], low, high);
  }
}

void nn_forward(NeuralNetwork nn) {
  for (size_t i = 0; i < nn.count; i++) {
    matrix_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
    matrix_acc(nn.as[i + 1], nn.bs[i]);
    matrix_sigf(nn.as[i + 1]);
  }
}

float nn_cost(NeuralNetwork nn, Matrix ti, Matrix to) {
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(ti.cols == NN_INPUT(nn).cols);
  NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

  size_t n = ti.rows;
  float cost = 0;
  for (size_t i = 0; i < n; i++) {
    Matrix x = matrix_row(ti, i);
    Matrix y = matrix_row(to, i);

    matrix_copy(NN_INPUT(nn), x);
    nn_forward(nn);

    size_t m = ti.cols;
    for (size_t j = 0; j < m; j++) {
      float diff = MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(y, 0, j);
      cost += diff * diff;
    }
  }
  return cost / n;
}

void nn_finite_diff(NeuralNetwork nn, NeuralNetwork grad, float eps, Matrix ti,
                    Matrix to) {
  float save;
  float cost = nn_cost(nn, ti, to);

#define NN_FINITE_DIFF(field)                                                  \
  {                                                                            \
    for (size_t j = 0; j < nn.field.rows; j++) {                               \
      for (size_t k = 0; k < nn.field.cols; k++) {                             \
        save = MATRIX_AT(nn.field, j, k);                                      \
        MATRIX_AT(nn.field, j, k) += eps;                                      \
        MATRIX_AT(grad.field, j, k) = (nn_cost(nn, ti, to) - cost) / eps;      \
        MATRIX_AT(nn.field, j, k) = save;                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  while (0)                                                                    \
    ;

  for (size_t i = 0; i < nn.count; i++) {
    NN_FINITE_DIFF(ws[i]);
    NN_FINITE_DIFF(bs[i]);
  }

#undef NN_FINITE_DIFF
}

void nn_learn(NeuralNetwork nn, NeuralNetwork grad, float rate) {

#define NN_LEARN(field)                                                        \
  {                                                                            \
    for (size_t j = 0; j < nn.field.rows; j++) {                               \
      for (size_t k = 0; k < nn.field.cols; k++) {                             \
        MATRIX_AT(nn.field, j, k) -= rate * MATRIX_AT(grad.field, j, k);       \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  while (0)                                                                    \
    ;

  for (size_t i = 0; i < nn.count; i++) {
    NN_LEARN(ws[i]);
    NN_LEARN(bs[i]);
  }

#undef NN_LEARN
}

#endif // NN_IMPLEMENTATION