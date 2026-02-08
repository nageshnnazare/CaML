#define NN_IMPLEMENTATION
#include "nn.h"

#include "raylib.h"

int main() {
  const int screenWidth = 800;
  const int screenHeight = 450;

  InitWindow(screenWidth, screenHeight, "Neural Network Visualizer");
  SetTargetFPS(60);

  size_t arch[] = {2, 2, 1};
  NN_NeuralNetwork nn = nn_alloc(arch, NN_ARRAY_LEN(arch));
  nn_rand(nn, 0.0, 1.0);
  NN_PRINT(nn);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawCircle(screenWidth / 2, screenHeight / 2, 100, RED);
    DrawText("Neural Network Visualizer", 10, 10, 20, DARKGRAY);
    EndDrawing();
  }

  nn_free(nn);
  CloseWindow();
  return 0;
}