#include <math.h>   // for expf
#include <stdio.h>  // for printf
#include <stdlib.h> // for rand
#include <time.h>   // for time

#define OR_EXAMPLE
// #define AND_EXAMPLE

float train[4][3] = {
// {x1, x2, y}
#ifdef OR_EXAMPLE
    // (OR function)
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
#endif
#ifdef AND_EXAMPLE
    // (AND function)
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
#endif
};
#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) { return ((float)rand() / (float)RAND_MAX); }

float sigmoidf(float x) { return (1.0f / (1.0f + expf(-x))); }

float cost_func(float w1, float w2, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float yp = sigmoidf(x1 * w1 + x2 * w2 + b);
    float loss = yp - y;
    result += loss * loss;
  }
  return result /= train_count;
}

int main(int argc, char *argv[]) {
  srand(time(0));
  float w1 = rand_float();
  float w2 = rand_float();
  float b = rand_float();

  float eps = 1e-1;
  float rate = 1e-1;

  for (int i = 0; i < 2000; ++i) {
    float dw1 = (cost_func(w1 + eps, w2, b) - cost_func(w1, w2, b)) / eps;
    float dw2 = (cost_func(w1, w2 + eps, b) - cost_func(w1, w2, b)) / eps;
    float db = (cost_func(w1, w2, b + eps) - cost_func(w1, w2, b)) / eps;

    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
  }

  printf("w1: %f, w2: %f, b:%f, f(0, 0) = %d\n", w1, w2, b,
         sigmoidf(w1 * 0 + w2 * 0 + b) < 0.5 ? 0 : 1);
  printf("w1: %f, w2: %f, b:%f, f(0, 1) = %d\n", w1, w2, b,
         sigmoidf(w1 * 0 + w2 * 1 + b) < 0.5 ? 0 : 1);
  printf("w1: %f, w2: %f, b:%f, f(1, 0) = %d\n", w1, w2, b,
         sigmoidf(w1 * 1 + w2 * 0 + b) < 0.5 ? 0 : 1);
  printf("w1: %f, w2: %f, b:%f, f(1, 1) = %d\n", w1, w2, b,
         sigmoidf(w1 * 1 + w2 * 1 + b) < 0.5 ? 0 : 1);

  return 0;
}
