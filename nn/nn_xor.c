/**
 * @file nn_xor.c
 * @brief Implement XOR gate using a 2-layer neural network with the Matrix library.
 *
 * This implementation uses the Matrix library to perform forward passes,
 * compute costs, and perform finite difference gradient approximation.
 */

#define NN_IMPLEMENTATION
#include "nn.h"

/**
 * @struct Xor
 * @brief Represents a simple 2-layer neural network for XOR.
 *
 * Architecture:
 * - Input (a0): 1x2 Matrix
 * - Hidden Layer (a1): 1x2 Matrix (Weights: w1 2x2, Bias: b1 1x2)
 * - Output Layer (a2): 1x1 Matrix (Weights: w2 2x1, Bias: b2 1x1)
 */
typedef struct {
  Matrix a0;         // Layer 0 (Input)
  Matrix w1, b1, a1; // Layer 1 (Hidden)
  Matrix w2, b2, a2; // Layer 2 (Output)
} Xor;

/**
 * @brief Allocates memory for the XOR network matrices.
 * @param nn Pointer to the Xor struct to initialize.
 */
void xor_alloc(Xor *nn) {
  nn->a0 = matrix_alloc(1, 2);

  nn->w1 = matrix_alloc(2, 2);
  nn->b1 = matrix_alloc(1, 2);
  nn->a1 = matrix_alloc(1, 2);

  nn->w2 = matrix_alloc(2, 1);
  nn->b2 = matrix_alloc(1, 1);
  nn->a2 = matrix_alloc(1, 1);
}

/**
 * @brief Frees memory allocated for the XOR network.
 * @param nn Pointer to the Xor struct to clean up.
 */
void xor_free(Xor *nn) {
  matrix_free(nn->a0);
  matrix_free(nn->w1);
  matrix_free(nn->b1);
  matrix_free(nn->a1);
  matrix_free(nn->w2);
  matrix_free(nn->b2);
  matrix_free(nn->a2);
}

/**
 * @brief Randomly initializes weights and biases for the network.
 * @param nn Pointer to the Xor struct.
 */
void xor_rand(Xor *nn) {
  matrix_rand(nn->w1, 0, 1);
  matrix_rand(nn->b1, 0, 1);
  matrix_rand(nn->w2, 0, 1);
  matrix_rand(nn->b2, 0, 1);
}

/**
 * @brief Performs a forward pass through the network.
 * @param nn Pointer to the Xor struct.
 */
void xor_forward(Xor *nn) {
  // a1 = sigmoid(a0 * w1 + b1)
  matrix_dot(nn->a1, nn->a0, nn->w1);
  matrix_acc(nn->a1, nn->b1);
  matrix_sigf(nn->a1);

  // a2 = sigmoid(a1 * w2 + b2)
  matrix_dot(nn->a2, nn->a1, nn->w2);
  matrix_acc(nn->a2, nn->b2);
  matrix_sigf(nn->a2);
}

/**
 * @brief Computes the Mean Squared Error cost for the given input/output data.
 * @param nn Pointer to the Xor struct.
 * @param ti Matrix of input training samples.
 * @param to Matrix of expected output training samples.
 * @return float Computed cost.
 */
float xor_cost(Xor *nn, Matrix ti, Matrix to) {
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(ti.cols == nn->a0.cols);
  NN_ASSERT(to.cols == nn->a2.cols);

  size_t n = ti.rows;
  float cost = 0;
  for (size_t i = 0; i < n; i++) {
    Matrix x = matrix_row(ti, i);
    Matrix y = matrix_row(to, i);

    matrix_copy(nn->a0, x);
    xor_forward(nn);

    size_t m = ti.cols;
    for (size_t j = 0; j < m; j++) {
      float diff = MATRIX_AT(nn->a2, 0, j) - MATRIX_AT(y, 0, j);
      cost += diff * diff;
    }
  }
  return cost / n;
}

/**
 * @brief Computes the gradient of the cost function using finite differences.
 * @param nn Pointer to the Xor struct.
 * @param grad Pointer to a Xor struct to store the computed gradient.
 * @param eps Finite difference step size.
 * @param ti Input training data.
 * @param to Output training data.
 */
void xor_finite_diff(Xor *nn, Xor *grad, float eps, Matrix ti, Matrix to) {
  float save;
  float cost = xor_cost(nn, ti, to);

#define FINITE_DIFF(field)                                                     \
  for (size_t i = 0; i < nn->field.rows; i++) {                                \
    for (size_t j = 0; j < nn->field.cols; j++) {                              \
      save = MATRIX_AT(nn->field, i, j);                                       \
      MATRIX_AT(nn->field, i, j) += eps;                                       \
      MATRIX_AT(grad->field, i, j) = (xor_cost(nn, ti, to) - cost) / eps;      \
      MATRIX_AT(nn->field, i, j) = save;                                       \
    }                                                                          \
  }

  FINITE_DIFF(w1);
  FINITE_DIFF(b1);
  FINITE_DIFF(w2);
  FINITE_DIFF(b2);
}

/**
 * @brief Updates the network weights and biases based on the gradient and learning rate.
 * @param nn Pointer to the Xor struct.
 * @param grad Pointer to the computed gradient.
 * @param rate Learning rate.
 */
void xor_learn(Xor *nn, Xor *grad, float rate) {

#define LEARN(field)                                                           \
  for (size_t i = 0; i < nn->field.rows; i++) {                                \
    for (size_t j = 0; j < nn->field.cols; j++) {                              \
      MATRIX_AT(nn->field, i, j) -= rate * MATRIX_AT(grad->field, i, j);       \
    }                                                                          \
  }

  LEARN(w1);
  LEARN(b1);
  LEARN(w2);
  LEARN(b2);
}

/** @brief XOR Training data: {x1, x2, y} */
float train_data[] = {
    0, 0, 0, 
    0, 1, 1, 
    1, 0, 1, 
    1, 1, 0,
};

int main() {
  Xor nn;
  xor_alloc(&nn);
  xor_rand(&nn);

  Xor grad;
  xor_alloc(&grad);

  float eps = 1e-1;
  float rate = 1e-1;

  size_t stride = 3;
  size_t n = sizeof(train_data) / sizeof(train_data[0]) / stride;

  Matrix ti = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .data = train_data,
  };

  Matrix to = {
      .rows = n,
      .cols = 1,
      .stride = stride,
      .data = train_data + 2,
  };

  // Training loop
  for (size_t i = 0; i < 100000; i++) {
    xor_finite_diff(&nn, &grad, eps, ti, to);
    xor_learn(&nn, &grad, rate);
  }

  // Verification pass
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      MATRIX_AT(nn.a0, 0, 0) = i;
      MATRIX_AT(nn.a0, 0, 1) = j;
      xor_forward(&nn);
      printf("%zu ^ %zu = %f\n", i, j, MATRIX_AT(nn.a2, 0, 0));
    }
  }

  xor_free(&nn);
  xor_free(&grad);

  return 0;
}