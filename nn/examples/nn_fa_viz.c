#define NN_IMPLEMENTATION
#include "../nn.h"
#include "raylib.h"
#include <time.h>

#define BITS 3

void nn_render_raylib(NN_NeuralNetwork nn) {
  int screenWidth = GetScreenWidth();
  int screenHeight = GetScreenHeight();

  Color background_color = {0x20, 0x20, 0x20, 0xff};
  Color foreground_color = {0xaa, 0xaa, 0xaa, 0xff};
  Color low_color = {0xee, 0x00, 0xee, 0xff};
  Color high_color = {0x00, 0xee, 0x00, 0xff};

  ClearBackground(background_color);
  DrawText("Neural Network Visualizer", 10, 0, 20, foreground_color);

  int neuron_radius = 20;
  int layer_border_vpad = 50;
  int layer_border_hpad = 50;
  int nn_width = screenWidth - layer_border_hpad * 2;
  int nn_height = screenHeight - layer_border_vpad * 2;
  int nn_x = screenWidth / 2 - nn_width / 2;
  int nn_y = screenHeight / 2 - nn_height / 2;
  size_t arch_count = NN_ARCH_COUNT(nn);
  int layer_hpad = nn_width / arch_count;

  for (size_t l = 0; l < arch_count; ++l) {
    int layer_vpad = nn_height / NN_ACTIVATION_LAYER_COL_COUNT(nn, l);
    for (size_t i = 0; i < NN_ACTIVATION_LAYER_COL_COUNT(nn, l); ++i) {
      int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
      int cy1 = nn_y + i * layer_vpad + layer_vpad / 2;

      if (l + 1 < arch_count) {
        int layer_vpad2 = nn_height / NN_ACTIVATION_LAYER_COL_COUNT(nn, l + 1);
        for (size_t j = 0; j < NN_ACTIVATION_LAYER_COL_COUNT(nn, l + 1); ++j) {
          int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
          int cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;

          high_color.a =
              floorf(255.f * sigmoidf(NN_MAT_AT(NN_WEIGHT_LAYER(nn, l), j, i)));
          Color conn_color = ColorAlphaBlend(low_color, high_color, WHITE);
          DrawCircleGradient(cx1, cy1, neuron_radius, conn_color, conn_color);
          DrawLine(cx1, cy1, cx2, cy2, conn_color);
        }
      }
      if (l > 0) {
        high_color.a = floorf(
            255.f * sigmoidf(NN_MAT_AT(NN_WEIGHT_LAYER(nn, l - 1), 0, i)));
        Color conn_color = ColorAlphaBlend(low_color, high_color, WHITE);
        DrawCircleGradient(cx1, cy1, neuron_radius, conn_color, conn_color);
      } else {
        DrawCircleGradient(cx1, cy1, neuron_radius, foreground_color,
                           foreground_color);
      }
    }
  }
}

int main() {
  srand(time(NULL));

  size_t n = (1 << BITS); // Number of possible values for BITS bits
  size_t rows = n * n;    // Total combinations of (x, y)

  // Input NN_Matrix: (x_bits, y_bits)
  NN_Matrix ti = nn_mat_alloc(rows, BITS * 2);
  // Output NN_Matrix: (sum_bits, overflow_bit)
  NN_Matrix to = nn_mat_alloc(rows, BITS + 1);

  // Generate training data for all possible sums of x and y
  for (size_t i = 0; i < ti.rows; i++) {
    size_t x = i / n;
    size_t y = i % n;
    size_t z = x + y;
    for (size_t j = 0; j < BITS; j++) {
      // Decompose x and y into bits for input
      NN_MAT_AT(ti, i, j) = (x >> j) & 1;
      NN_MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
      // Decompose z (sum) into bits for target output
      NN_MAT_AT(to, i, j) = (z >> j) & 1;
    }
    // Set the overflow bit (carry out of the last bit)
    NN_MAT_AT(to, i, BITS) = z >= n;
  }

  // Define the network architecture: [Inputs, Hidden Layer, Outputs]
  size_t arch[] = {2 * BITS, 2 * BITS - 1, BITS + 1};
  NN_NeuralNetwork nn = nn_alloc(arch, NN_ARRAY_LEN(arch));
  NN_NeuralNetwork grad = nn_alloc(arch, NN_ARRAY_LEN(arch));

  // Randomize initial weights
  nn_rand(nn, 0.0, 1.0);
  float rate = 1;

  const int screenWidth = 800;
  const int screenHeight = 600;

  InitWindow(screenWidth, screenHeight, "Neural Network Visualizer");
  SetTargetFPS(600);

  size_t i = 0;
  while (!WindowShouldClose()) {
    if (i < 20000) {
      nn_backprop(nn, grad, ti, to);
      nn_learn(nn, grad, rate);
      i++;
      printf("%zu: Cost: %f\n", i, nn_cost(nn, ti, to));
    }

    BeginDrawing();
    nn_render_raylib(nn);
    EndDrawing();
  }

  // Verification: Test the trained network against all inputs
  size_t fail_count = 0;
  for (size_t x = 0; x < n; x++) {
    for (size_t y = 0; y < n; y++) {
      size_t z = x + y;
      // Load current x and y into the NN input layer
      for (size_t j = 0; j < BITS; j++) {
        NN_MAT_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
        NN_MAT_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
      }

      nn_forward(nn);

      // Check overflow bit
      if (NN_MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5) {
        if (z < n) {
          printf("%zu + %zu = (ov<>%zu)\n", x, y, z);
          fail_count++;
        }
      } else {
        // Reconstruct numeric sum from output bits
        size_t a = 0;
        for (size_t j = 0; j < BITS; j++) {
          size_t bit = NN_MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5;
          a |= bit << j;
        }
        // Verify reconstructed sum against actual sum
        if (a != z) {
          printf("%zu + %zu = (%zu<>%zu)\n", x, y, z, a);
          fail_count++;
        }
      }
    }
  }

  printf("Fail count: %zu\n", fail_count);

  CloseWindow();

  // Cleanup
  nn_free(nn);
  nn_free(grad);
  nn_mat_free(ti);
  nn_mat_free(to);

  return 0;
}