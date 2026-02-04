/**
 * @file nn.h
 * @brief A lightweight, header-only neural network library in C.
 *
 * This library provides a modular implementation of a multi-layer perceptron
 * (MLP). It uses a custom Matrix library for all linear algebra operations.
 *
 * @section usage Usage
 * To use the implementation in a single C file, define `NN_IMPLEMENTATION`
 * before including this header:
 * @code
 * #define NN_IMPLEMENTATION
 * #include "nn.h"
 * @endcode
 *
 * @section concepts Machine Learning Fundamentals
 * @subsection linear_transformation Linear Transformation: y = wx + b
 * The core of each neuron is a linear transformation:
 * - **w (Weights)**: Parameters that determine the strength of connection
 * between neurons.
 * - **x (Inputs)**: Features or activations from the previous layer.
 * - **b (Biases)**: Parameters that allow the neuron to shift the activation
 * function.
 * - **y (Logits)**: The raw output before applying a non-linear activation
 * function.
 *
 * @subsection forward_pass Forward Propagation
 * Computes the network's output by performing matrix multiplication (dot
 * product) followed by adding biases and applying a non-linear activation
 * (Sigmoid).
 *
 * @subsection cost_function Cost/Loss Function
 * Measures the error between predicted and target values (Mean Squared Error).
 *
 * @subsection finite_diff Finite Differences Gradient Approximation
 * Approximates gradients by perturbing each parameter (Weight or Bias) by a
 * small epsilon ($\epsilon$) and observing the change in cost.
 *
 * @subsection gradient_descent Gradient Descent
 * Optimizes the network by updating parameters in the opposite direction of the
 * gradient, scaled by a learning rate.
 */

#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdio.h>

/**
 * @brief Generates a random float between 0.0 and 1.0.
 * @return float A random value for weight/bias initialization.
 */
float randf(void);

/**
 * @brief Standard Logistic Sigmoid activation function.
 *
 * Maps any real-valued number into the (0, 1) range.
 * Formula: f(x) = 1 / (1 + exp(-x))
 *
 * @param x The logit value.
 * @return float The non-linear activation.
 */
float sigmoidf(float x);

/**
 * @brief Helper macro for array size calculation.
 */
#define NN_ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * @brief Memory allocation wrapper.
 */
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC(size) malloc(size)
#endif // NN_MALLOC

/**
 * @brief Memory deallocation wrapper.
 */
#ifndef NN_FREE
#include <stdlib.h>
#define NN_FREE(ptr) free(ptr)
#endif // NN_FREE

/**
 * @brief Internal assertion macro for safe dimensions.
 */
#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT(condition) assert(condition)
#endif // NN_ASSERT

/**
 * @struct Matrix
 * @brief A generic 2D matrix structure.
 *
 * Uses a flat float array for memory efficiency.
 */
typedef struct {
  size_t rows;   /**< Number of examples or features. */
  size_t cols;   /**< Number of features or outputs. */
  size_t stride; /**< Step size for sub-matrix memory access. */
  float *data;   /**< Linearized buffer of weights, biases, or activations. */
} Matrix;

/**
 * @brief Macro to access matrix elements given row i and column j.
 */
#define MATRIX_AT(m, i, j) ((m).data[(i) * (m).stride + (j)])

/**
 * @brief Interface to print a named matrix to stdout.
 */
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)

/**
 * @brief Allocates a matrix with specified dimensions.
 * @param rows Rows count.
 * @param cols Columns count.
 * @return Matrix The initialized matrix structure.
 */
Matrix matrix_alloc(size_t rows, size_t cols);

/**
 * @brief Frees matrix memory buffer.
 * @param m Matrix to deallocate.
 */
void matrix_free(Matrix m);

/**
 * @brief Pretty-prints a matrix with unicode borders.
 * @param m Matrix to print.
 * @param name Variable identifier.
 * @param padding Indentation level.
 */
void matrix_print(Matrix m, const char *name, int padding);

/**
 * @brief Randomly initializes matrix elements.
 * @param m Target matrix.
 * @param low Minimum value.
 * @param high Maximum value.
 */
void matrix_rand(Matrix m, float low, float high);

/**
 * @brief Fills the entire matrix with a constant scalar.
 * @param m Target matrix.
 * @param val Scalar value.
 */
void matrix_fill(Matrix m, float val);

/**
 * @brief Zeros out the entire matrix.
 * @param m Target matrix.
 */
void matrix_zero(Matrix m);

/**
 * @brief Returns a view of a single row as a (1 x cols) matrix.
 * @param m Source matrix.
 * @param row Row index.
 * @return Matrix A shallow copy (view) of the row data.
 */
Matrix matrix_row(Matrix m, size_t row);

/**
 * @brief Returns a rectangular sub-section of a larger matrix.
 * @param m Source matrix.
 * @param row Start row.
 * @param col Start column.
 * @param rows Sub-rows count.
 * @param cols Sub-cols count.
 * @return Matrix A shallow copy (view) with adjusted stride.
 */
Matrix matrix_submatrix(Matrix m, size_t row, size_t col, size_t rows,
                        size_t cols);

/**
 * @brief Deep copies contents of matrix 'a' into 'res'.
 * @param res Destination matrix.
 * @param a Source matrix.
 */
void matrix_copy(Matrix res, Matrix a);

/**
 * @brief Applies element-wise sigmoid activation in-place.
 */
void matrix_sigf(Matrix m);

/**
 * @brief Element-wise accumulation: res += a.
 */
void matrix_acc(Matrix res, Matrix a);

/**
 * @brief Element-wise addition: res = a + b.
 */
void matrix_sum(Matrix res, Matrix a, Matrix b);

/**
 * @brief Matrix dot product (Linear transformation): res = a * b.
 *
 * Implements the core wx multiplication in the MLP hidden layers.
 */
void matrix_dot(Matrix res, Matrix a, Matrix b);

/**
 * @struct NeuralNetwork
 * @brief MLP Architecture descriptor and parameter container.
 */
typedef struct {
  size_t count; /**< Number of weight/bias layers. */
  Matrix *ws;   /**< Weight matrices for each layer connection. */
  Matrix *bs;   /**< Bias vectors for each layer. */
  Matrix *as;   /**< Activation vectors for each layer (including input). */
} NeuralNetwork;

/**
 * @brief Interface to print the entire model architecture and parameters.
 */
#define NN_PRINT(nn) nn_print(nn, #nn)

/**
 * @brief Shallow access to the input layer activations.
 */
#define NN_INPUT(nn) ((nn).as[0])

/**
 * @brief Shallow access to the final output layer activations.
 */
#define NN_OUTPUT(nn) ((nn).as[(nn).count])

/**
 * @brief Initializes a Neural Network model from an architecture array.
 * @param arch Array of neurons per layer (e.g., {2, 2, 1}).
 * @param arch_count Length of arch array.
 * @return NeuralNetwork The allocated model.
 */
NeuralNetwork nn_alloc(size_t *arch, size_t arch_count);

/**
 * @brief Deallocates all weights, biases, and activation buffers in the model.
 */
void nn_free(NeuralNetwork nn);

/**
 * @brief Displays the network's layers, weights, and current activations.
 */
void nn_print(NeuralNetwork nn, const char *name);

/**
 * @brief Performs random initialization of model parameters.
 */
void nn_rand(NeuralNetwork nn, float low, float high);

/**
 * @brief Zero initialization of model parameters.
 */
void nn_zero(NeuralNetwork nn);

/**
 * @brief Forward Propagation: x -> [Linear + Activation] -> y
 *
 * Computes activations for all layers given the current input in NN_INPUT(nn).
 */
void nn_forward(NeuralNetwork nn);

/**
 * @brief Mean Squared Error (MSE) Cost calculation.
 *
 * Sum of squared differences between predictions (from forward pass) and
 * targets.
 * @param nn The model to evaluate.
 * @param ti Training input matrix.
 * @param to Training target matrix.
 * @return float Normalized scalar cost.
 */
float nn_cost(NeuralNetwork nn, Matrix ti, Matrix to);

/**
 * @brief Finite Differences Gradient Approximation.
 *
 * Approximates partial derivatives of the cost function with respect to every
 * weight and bias in the network.
 *
 * @param nn The model.
 * @param grad Gradient container (must match nn architecture).
 * @param eps Small perturbation value (epsilon).
 * @param ti Training inputs.
 * @param to Training targets.
 */
void nn_finite_diff(NeuralNetwork nn, NeuralNetwork grad, float eps, Matrix ti,
                    Matrix to);

/**
 * @brief Backpropagation Algorithm.
 *
 * Computes the gradient of the cost function with respect to every weight and
 * bias in the network.
 *
 * @param nn The model.
 * @param grad Gradient container (must match nn architecture).
 * @param ti Training inputs.
 * @param to Training targets.
 */
void nn_backprop(NeuralNetwork nn, NeuralNetwork grad, Matrix ti, Matrix to);

/**
 * @brief Stochastic Gradient Descent (SGD) update step.
 *
 * Updates parameters: w = w - rate * grad
 *
 * @param nn Model to optimize.
 * @param grad Pre-calculated gradient.
 * @param rate Learning rate (Step size).
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

void matrix_zero(Matrix m) {
  matrix_fill(m, 0);
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

void nn_zero(NeuralNetwork nn) {
  for (size_t i = 0; i < nn.count; i++) {
    matrix_zero(nn.ws[i]);
    matrix_zero(nn.bs[i]);
    matrix_zero(nn.as[i]);
  }
  matrix_zero(nn.as[nn.count]);
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

void nn_backprop(NeuralNetwork nn, NeuralNetwork grad, Matrix ti, Matrix to) {
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(ti.cols == NN_INPUT(nn).cols);
  NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

  size_t n = ti.rows;
  nn_zero(grad);

  // i = current training sample
  // l = current layer
  // j = current activation
  // k = previous weight

  for (size_t i = 0; i < n; i++) {
    matrix_copy(NN_INPUT(nn), matrix_row(ti, i));
    nn_forward(nn);

    for (size_t j = 0; j <= nn.count; j++) {
      matrix_zero(grad.as[j]);
    }

    for (size_t j = 0; j < to.cols; j++) {
      MATRIX_AT(NN_OUTPUT(grad), 0, j) =
          MATRIX_AT(NN_OUTPUT(nn), 0, j) - MATRIX_AT(to, i, j);
    }

    for (size_t l = nn.count; l > 0; l--) {
      for (size_t j = 0; j < nn.as[l].cols; j++) {
        float a = MATRIX_AT(nn.as[l], 0, j);
        float da = MATRIX_AT(grad.as[l], 0, j);
        MATRIX_AT(grad.bs[l - 1], 0, j) += 2 * da * a * (1 - a);
        for (size_t k = 0; k < nn.as[l - 1].cols; k++) {
          // j = weight matrix column
          // k = weight matrix row
          float prev_a = MATRIX_AT(nn.as[l - 1], 0, k);
          float w = MATRIX_AT(nn.ws[l - 1], k, j);
          MATRIX_AT(grad.ws[l - 1], k, j) += 2 * da * a * (1 - a) * prev_a;
          MATRIX_AT(grad.as[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
        }
      }
    }
  }
  for (size_t i = 0; i < grad.count; ++i) {
    for (size_t j = 0; j < grad.ws[i].rows; ++j) {
      for (size_t k = 0; k < grad.ws[i].cols; ++k) {
        MATRIX_AT(grad.ws[i], j, k) /= n;
      }
    }
    for (size_t j = 0; j < grad.bs[i].rows; ++j) {
      for (size_t k = 0; k < grad.bs[i].cols; ++k) {
        MATRIX_AT(grad.bs[i], j, k) /= n;
      }
    }
  }
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