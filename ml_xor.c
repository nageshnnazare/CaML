#include <math.h>   // for expf
#include <stdio.h>  // for printf
#include <stdlib.h> // for rand
#include <time.h>   // for time

typedef float sample[3];

sample xor_train[] = {
    // {x1, x2, y}
    // (XOR function)
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0},
    // xor = (x | y) & ~(x & y)
};
sample or_train[] = {
    // {x1, x2, y}
    // (OR function)
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
    // or = (x | y)
};
sample and_train[] = {
    // {x1, x2, y}
    // (AND function)
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
    // and = (x & y)
};
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

float rand_float(void) { return ((float)rand() / (float)RAND_MAX); }

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

float forward_pass(Xor model, float x1, float x2) {
  float a = sigmoidf(model.or_w1 * x1 + model.or_w2 * x2 + model.or_b);
  float b = sigmoidf(model.nand_w1 * x1 + model.nand_w2 * x2 + model.nand_b);
  return sigmoidf(model.and_w1 * a + model.and_w2 * b + model.and_b);
}

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

Xor finite_diff(Xor model, float eps) {
  Xor new_model;
  float cost = cost_func(model);
  float save;

  save = model.or_w1;
  model.or_w1 += eps;
  new_model.or_w1 = (cost_func(model) - cost) / eps;
  model.or_w1 = save;

  save = model.or_w2;
  model.or_w2 += eps;
  new_model.or_w2 = (cost_func(model) - cost) / eps;
  model.or_w2 = save;

  save = model.or_b;
  model.or_b += eps;
  new_model.or_b = (cost_func(model) - cost) / eps;
  model.or_b = save;

  save = model.nand_w1;
  model.nand_w1 += eps;
  new_model.nand_w1 = (cost_func(model) - cost) / eps;
  model.nand_w1 = save;

  save = model.nand_w2;
  model.nand_w2 += eps;
  new_model.nand_w2 = (cost_func(model) - cost) / eps;
  model.nand_w2 = save;

  save = model.nand_b;
  model.nand_b += eps;
  new_model.nand_b = (cost_func(model) - cost) / eps;
  model.nand_b = save;

  save = model.and_w1;
  model.and_w1 += eps;
  new_model.and_w1 = (cost_func(model) - cost) / eps;
  model.and_w1 = save;

  save = model.and_w2;
  model.and_w2 += eps;
  new_model.and_w2 = (cost_func(model) - cost) / eps;
  model.and_w2 = save;

  save = model.and_b;
  model.and_b += eps;
  new_model.and_b = (cost_func(model) - cost) / eps;
  model.and_b = save;

  return new_model;
}

Xor apply_diff(Xor model, Xor new_model, float rate) {
  model.or_w1 -= rate * new_model.or_w1;
  model.or_w2 -= rate * new_model.or_w2;
  model.or_b -= rate * new_model.or_b;
  model.nand_w1 -= rate * new_model.nand_w1;
  model.nand_w2 -= rate * new_model.nand_w2;
  model.nand_b -= rate * new_model.nand_b;
  model.and_w1 -= rate * new_model.and_w1;
  model.and_w2 -= rate * new_model.and_w2;
  model.and_b -= rate * new_model.and_b;
  return model;
}

int main(int argc, char *argv[]) {
  srand(time(0));
  {
    train = xor_train;
    Xor model = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 20000; ++i) {
      Xor new_model = finite_diff(model, eps);
      model = apply_diff(model, new_model, rate);
      // printf("cost = %f\n", cost_func(model));
    }

    // print_xor(model);

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        printf("%d ^ %d = %d\n", i, j, forward_pass(model, i, j) < 0.5 ? 0 : 1);
      }
    }
    printf("------------------\n");
  }
  {
    train = or_train;
    Xor model = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 20000; ++i) {
      Xor new_model = finite_diff(model, eps);
      model = apply_diff(model, new_model, rate);
      // printf("cost = %f\n", cost_func(model));
    }

    // print_xor(model);

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        printf("%d | %d = %d\n", i, j, forward_pass(model, i, j) < 0.5 ? 0 : 1);
      }
    }
    printf("------------------\n");
  }
  {
    train = and_train;
    Xor model = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 20000; ++i) {
      Xor new_model = finite_diff(model, eps);
      model = apply_diff(model, new_model, rate);
      // printf("cost = %f\n", cost_func(model));
    }

    // print_xor(model);

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        printf("%d & %d = %d\n", i, j, forward_pass(model, i, j) < 0.5 ? 0 : 1);
      }
    }
    printf("------------------\n");
  }
  {
    train = nand_train;
    Xor model = rand_xor();

    float eps = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 20000; ++i) {
      Xor new_model = finite_diff(model, eps);
      model = apply_diff(model, new_model, rate);
      // printf("cost = %f\n", cost_func(model));
    }

    // print_xor(model);

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        printf("%d ~& %d = %d\n", i, j, forward_pass(model, i, j) < 0.5 ? 0 : 1);
      }
    }
    printf("------------------\n");
  }

  return 0;
}
