/**
 * @file ml_helloWorld.c
 * @brief Simple Linear Regression (y = w * x + b) using Finite Differences.
 *
 * This example demonstrates the most basic form of machine learning:
 * finding a line that fits a set of data points.
 */

#include <stdio.h>  // for printf
#include <stdlib.h> // for rand
#include <time.h>   // for time

/**
 * @brief Training data: {input, expected_output}.
 * The model should learn y = 2x.
 */
float train[][2] = {
    // {x, y}
    {0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8},
};
#define train_count (sizeof(train) / sizeof(train[0]))

/**
 * @brief Generates a random float between 0.0 and 1.0.
 */
float rand_float(void) { return ((float)rand() / (float)RAND_MAX); }

/**
 * @brief Computes the Mean Squared Error (Loss) for the current model.
 *
 * y = w * x + b
 * loss = average of (prediction - target)^2
 *
 * Given a weight and bias, compute the average loss
 * for our model y = w * x + b, by iterating over
 * our input data
 *
 * @param w Current weight.
 * @param b Current bias.
 * @return float Average loss.
 */
float cost_func(float w, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x = train[i][0]; // input
    float y = train[i][1]; // expected output
    // y = w * x + b;
    float yp = x * w + b;  // prediction
    float loss = yp - y;   // error
    result += loss * loss; // squared error
  }
  return result /= train_count; // error per sample
}

int main(int argc, char *argv[]) {
  srand(time(0));
  float weight = rand_float();
  float bias = rand_float();

  float epsilon = 1e-3;       // Step size for calculating derivative
  float learning_rate = 1e-3; // How much to update weights/bias

  // Training loop
  for (int i = 0; i < 1000; ++i) {
    // Finite difference approximation of gradients
    // d = (f(a + h) - f(a)) / h;
    // d/dw cost_func ≈ (cost_func(w + eps) - cost_func(w)) / eps
    float dw =
        (cost_func(weight + epsilon, bias) - cost_func(weight, bias)) / epsilon;
    float db =
        (cost_func(weight, bias + epsilon) - cost_func(weight, bias)) / epsilon;

    // Gradient descent step
    weight -= dw * learning_rate;
    bias -= db * learning_rate;
  }

  printf("Model: y = %f * x + %f\n", weight, bias);
  printf("Target: y = 2.0 * x + 0.0\n");
  printf("Testing f(5): Expected 10.0, Got %f\n", (5 * weight + bias));

  return 0;
}
