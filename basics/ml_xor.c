/**
 * @file ml_xor.c
 * @brief Training a Multi-Layer Perceptron (MLP) to solve the XOR problem.
 *
 * XOR cannot be solved by a single neuron because it is not linearly separable.
 * This example uses a network with 2 hidden neurons (OR and NAND gates)
 * feeding into an AND gate.
 */

#include <math.h>   // for expf
#include <stdio.h>  // for printf
#include <stdlib.h> // for rand
#include <time.h>   // for time

typedef float sample[3];

/** @brief XOR Training data */
sample xor_train[] = {
    // {x1, x2, y}
    // (XOR function)
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
    // xor = (x | y) & ~(x & y)
};
/** @brief OR Training data (for comparison) */
sample or_train[] = {
    // {x1, x2, y}
    // (OR function)
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
    // or = (x | y)
};
/** @brief AND Training data (for comparison) */
sample and_train[] = {
    // {x1, x2, y}
    // (AND function)
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
    // and = (x & y)
};
/** @brief NAND Training data (for comparison) */
sample nand_train[] = {
    // {x1, x2, y}
    // (NAND function)
    {0, 0, 1},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
    // nand = ~(x & y)
};

sample *train = xor_train;
size_t train_count = 4;

/**
 * @struct Xor
 * @brief Represents the weights and biases for the entire MLP.
 * High-level architecture:
 * Input -> (OR Gate neuron, NAND Gate neuron) -> AND Gate neuron -> Output
 */
typedef struct {
  float or_w1;
  float or_w2;
  float or_b;
  float nand_w1;
  float nand_w2;
  float nand_b;
  float and_w1;
  float and_w2;
  float and_b;
} Xor;

/** @brief Generates a random float between 0.0 and 1.0. */
float rand_float(void) { return ((float)rand() / (float)RAND_MAX); }

/** @brief Initializes the network with random weights and biases. */
Xor rand_xor(void) {
  Xor model;
  model.or_w1 = rand_float();
  model.or_w2 = rand_float();
  model.or_b = rand_float();
  model.nand_w1 = rand_float();
  model.nand_w2 = rand_float();
  model.nand_b = rand_float();
  model.and_w1 = rand_float();
  model.and_w2 = rand_float();
  model.and_b = rand_float();
  return model;
}

void print_xor(Xor model) {
  printf("  or_w1 = %f\n", model.or_w1);
  printf("  or_w2 = %f\n", model.or_w2);
  printf("  or_b = %f\n", model.or_b);
  printf("  nand_w1 = %f\n", model.nand_w1);
  printf("  nand_w2 = %f\n", model.nand_w2);
  printf("  nand_b = %f\n", model.nand_b);
  printf("  and_w1 = %f\n", model.and_w1);
  printf("  and_w2 = %f\n", model.and_w2);
  printf("  and_b = %f\n", model.and_b);
}

float sigmoidf(float x) { return (1.0f / (1.0f + expf(-x))); }

/**
 * @brief Performs a forward pass through the network.
 */
float forward_pass(Xor model, float x1, float x2) {
  float a = sigmoidf(model.or_w1 * x1 + model.or_w2 * x2 + model.or_b);
  float b = sigmoidf(model.nand_w1 * x1 + model.nand_w2 * x2 + model.nand_b);
  return sigmoidf(model.and_w1 * a + model.and_w2 * b + model.and_b);
}

/** @brief Computes the Mean Squared Error (Loss). */
float cost_func(Xor model) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float yp = forward_pass(model, x1, x2);
    float loss = yp - y;
    result += loss * loss;
  }
  return result /= train_count;
}

/**
 * @brief Approximates the gradient using finite differences.
 * @param model Current model state.
 * @param eps Step size for differentiation.
 * @return Xor A "model" where each field contains the partial derivative of the
 * cost.
 */
Xor finite_diff(Xor model, float eps) {
  Xor d;
  float cost = cost_func(model);
  float save;

#define DIFF(field)                                                            \
  save = model.field;                                                          \
  model.field += eps;                                                          \
  d.field = (cost_func(model) - cost) / eps;                                   \
  model.field = save;

  DIFF(or_w1);
  DIFF(or_w2);
  DIFF(or_b);
  DIFF(nand_w1);
  DIFF(nand_w2);
  DIFF(nand_b);
  DIFF(and_w1);
  DIFF(and_w2);
  DIFF(and_b);

  return d;
}

/** @brief Updates the model weights and biases using the calculated gradient.
 */
Xor apply_diff(Xor model, Xor d, float rate) {
  model.or_w1 -= rate * d.or_w1;
  model.or_w2 -= rate * d.or_w2;
  model.or_b -= rate * d.or_b;
  model.nand_w1 -= rate * d.nand_w1;
  model.nand_w2 -= rate * d.nand_w2;
  model.nand_b -= rate * d.nand_b;
  model.and_w1 -= rate * d.and_w1;
  model.and_w2 -= rate * d.and_w2;
  model.and_b -= rate * d.and_b;
  return model;
}

/**
 * @brief Main function trains the network for XOR and other gates.
 */
int main(int argc, char *argv[]) {
  srand(time(0));

  sample *datasets[] = {xor_train, or_train, and_train, nand_train};
  const char *names[] = {"XOR", "OR", "AND", "NAND"};

  for (size_t d = 0; d < 4; ++d) {
    printf("Training for %s gate...\n", names[d]);
    train = datasets[d];
    Xor model = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 20000; ++i) {
      Xor g = finite_diff(model, eps);
      model = apply_diff(model, g, rate);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        printf("%d op %d = %d\n", i, j,
               forward_pass(model, i, j) < 0.5 ? 0 : 1);
      }
    }
    printf("------------------\n");
  }

  return 0;
}
