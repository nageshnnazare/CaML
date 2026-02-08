/**
 * @file mat_xor.c
 * @brief Implement XOR gate using a 2-layer neural network with the NN_Matrix
 * library.
 *
 * This implementation uses the NN_Matrix library to perform forward passes,
 * compute costs, and perform finite difference gradient approximation.
 */

#define NN_IMPLEMENTATION
#include "../nn.h"

#include <time.h>

/**
 * @struct Xor
 * @brief Represents a simple 2-layer neural network for XOR.
 *
 * Architecture:
 * - Input (a0): 1x2 NN_Matrix
 * - Hidden Layer (a1): 1x2 NN_Matrix (Weights: w1 2x2, Bias: b1 1x2)
 * - Output Layer (a2): 1x1 NN_Matrix (Weights: w2 2x1, Bias: b2 1x1)
 */
typedef struct {
  NN_Matrix a0;         // Layer 0 (Input)
  NN_Matrix w1, b1, a1; // Layer 1 (Hidden)
  NN_Matrix w2, b2, a2; // Layer 2 (Output)
} Xor;

/**
 * @brief Allocates memory for the XOR network matrices.
 * @param nn Pointer to the Xor struct to initialize.
 */
void xor_alloc(Xor *nn) {
  nn->a0 = nn_mat_alloc(1, 2);

  nn->w1 = nn_mat_alloc(2, 2);
  nn->b1 = nn_mat_alloc(1, 2);
  nn->a1 = nn_mat_alloc(1, 2);

  nn->w2 = nn_mat_alloc(2, 1);
  nn->b2 = nn_mat_alloc(1, 1);
  nn->a2 = nn_mat_alloc(1, 1);
}

/**
 * @brief Frees memory allocated for the XOR network.
 * @param nn Pointer to the Xor struct to clean up.
 */
void xor_free(Xor *nn) {
  nn_mat_free(nn->a0);
  nn_mat_free(nn->w1);
  nn_mat_free(nn->b1);
  nn_mat_free(nn->a1);
  nn_mat_free(nn->w2);
  nn_mat_free(nn->b2);
  nn_mat_free(nn->a2);
}

/**
 * @brief Randomly initializes weights and biases for the network.
 * @param nn Pointer to the Xor struct.
 */
void xor_rand(Xor *nn) {
  nn_mat_rand(nn->w1, 0, 1);
  nn_mat_rand(nn->b1, 0, 1);
  nn_mat_rand(nn->w2, 0, 1);
  nn_mat_rand(nn->b2, 0, 1);
}

/**
 * @brief Performs a forward pass through the network.
 * @param nn Pointer to the Xor struct.
 */
void xor_forward(Xor *nn) {
  // a1 = sigmoid(a0 * w1 + b1)
  nn_mat_dot(nn->a1, nn->a0, nn->w1);
  nn_mat_acc(nn->a1, nn->b1);
  nn_mat_sigf(nn->a1);

  // a2 = sigmoid(a1 * w2 + b2)
  nn_mat_dot(nn->a2, nn->a1, nn->w2);
  nn_mat_acc(nn->a2, nn->b2);
  nn_mat_sigf(nn->a2);
}

/**
 * @brief Computes the Mean Squared Error cost for the given input/output data.
 * @param nn Pointer to the Xor struct.
 * @param ti NN_Matrix of input training samples.
 * @param to NN_Matrix of expected output training samples.
 * @return float Computed cost.
 */
float xor_cost(Xor *nn, NN_Matrix ti, NN_Matrix to) {
  NN_ASSERT(ti.rows == to.rows);
  NN_ASSERT(ti.cols == nn->a0.cols);
  NN_ASSERT(to.cols == nn->a2.cols);

  size_t n = ti.rows;
  float cost = 0;
  for (size_t i = 0; i < n; i++) {
    NN_Matrix x = nn_mat_row(ti, i);
    NN_Matrix y = nn_mat_row(to, i);

    nn_mat_copy(nn->a0, x);
    xor_forward(nn);

    size_t m = ti.cols;
    for (size_t j = 0; j < m; j++) {
      float diff = NN_MAT_AT(nn->a2, 0, j) - NN_MAT_AT(y, 0, j);
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
void xor_finite_diff(Xor *nn, Xor *grad, float eps, NN_Matrix ti,
                     NN_Matrix to) {
  float save;
  float cost = xor_cost(nn, ti, to);

#define FINITE_DIFF(field)                                                     \
  for (size_t i = 0; i < nn->field.rows; i++) {                                \
    for (size_t j = 0; j < nn->field.cols; j++) {                              \
      save = NN_MAT_AT(nn->field, i, j);                                       \
      NN_MAT_AT(nn->field, i, j) += eps;                                       \
      NN_MAT_AT(grad->field, i, j) = (xor_cost(nn, ti, to) - cost) / eps;      \
      NN_MAT_AT(nn->field, i, j) = save;                                       \
    }                                                                          \
  }

  FINITE_DIFF(w1);
  FINITE_DIFF(b1);
  FINITE_DIFF(w2);
  FINITE_DIFF(b2);
}

/**
 * @brief Updates the network weights and biases based on the gradient and
 * learning rate.
 * @param nn Pointer to the Xor struct.
 * @param grad Pointer to the computed gradient.
 * @param rate Learning rate.
 */
void xor_learn(Xor *nn, Xor *grad, float rate) {

#define LEARN(field)                                                           \
  for (size_t i = 0; i < nn->field.rows; i++) {                                \
    for (size_t j = 0; j < nn->field.cols; j++) {                              \
      NN_MAT_AT(nn->field, i, j) -= rate * NN_MAT_AT(grad->field, i, j);       \
    }                                                                          \
  }

  LEARN(w1);
  LEARN(b1);
  LEARN(w2);
  LEARN(b2);
}

/** @brief XOR Training data: {x1, x2, y} */
float train_data[] = {
    0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
};

int main() {
  srand(time(NULL));
  Xor nn;
  xor_alloc(&nn);
  xor_rand(&nn);

  Xor grad;
  xor_alloc(&grad);

  float eps = 1e-1;
  float rate = 1e-1;

  size_t stride = 3;
  size_t n = sizeof(train_data) / sizeof(train_data[0]) / stride;

  NN_Matrix ti = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .data = train_data,
  };

  NN_Matrix to = {
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
      NN_MAT_AT(nn.a0, 0, 0) = i;
      NN_MAT_AT(nn.a0, 0, 1) = j;
      xor_forward(&nn);
      printf("%zu ^ %zu = %f\n", i, j, NN_MAT_AT(nn.a2, 0, 0));
    }
  }

  xor_free(&nn);
  xor_free(&grad);

  return 0;
}