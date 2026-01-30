#include <stdio.h>  // for printf
#include <stdlib.h> // for rand
#include <time.h>   // for time

float train[][2] = {
    // {x, y}
    {0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8},
};
#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) { return ((float)rand() / (float)RAND_MAX); }

// Given a weight and bias, compute the average loss
// for our model y = w * x + b, by iterating over
// our input data
float cost_func(float w, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x = train[i][0]; // input
    float y = train[i][1]; // expected output
    // y = w * x + b;
    float yp = x * w + b;  // prediction
    float loss = yp - y;   // loss
    result += loss * loss; // accumulate losses
  }
  return result /= train_count; // loss per sample
}

int main(int argc, char *argv[]) {
  srand(time(0));
  float weight = rand_float();
  float bias = rand_float();

  float epsilon = 1e-3;       // input step_size
  float learning_rate = 1e-3; // output step_size

  for (int i = 0; i < 1000; ++i) {
    // d = (f(a + h) - f(a)) / h; // finite difference derivate
    float dw =
        (cost_func(weight + epsilon, bias) - cost_func(weight, bias)) / epsilon;
    float db =
        (cost_func(weight, bias + epsilon) - cost_func(weight, bias)) / epsilon;
    weight -= dw * learning_rate; // update weight
    bias -= db * learning_rate;   // update bias
  }
  printf("w: %f, b: %f, f(5) = %f\n", weight, bias, (5 * weight + bias));

  return 0;
}
