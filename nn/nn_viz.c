#define NN_IMPLEMENTATION
#include "nn.h"
#define SV_IMPLEMENTATION
#include "deps/sv.h"

#include "raylib.h"
#include <time.h>

typedef struct {
  size_t count;
  size_t capacity;
  size_t *items;
} Arch;

#define DA_INIT_CAPACITY 128
#define da_append(da, item)                                                    \
  do {                                                                         \
    if ((da).count >= (da).capacity) {                                         \
      (da).capacity =                                                          \
          (da).capacity == 0 ? DA_INIT_CAPACITY : (da).capacity * 2;           \
      (da).items = realloc((da).items, (da).capacity * sizeof(*(da).items));   \
    }                                                                          \
    (da).items[(da).count++] = (item);                                         \
  } while (0)

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

  int buffer_len = 0;
  unsigned char *buffer = LoadFileData("nn/adder.arch", &buffer_len);

  String_View content = sv_from_parts((const char *)buffer, buffer_len);
  content = sv_trim_left(content);
  Arch arch = {0};

  while (content.count > 0 && isdigit(content.data[0])) {
    uint64_t val = sv_chop_u64(&content);
    da_append(arch, val);
    content = sv_trim_left(content);
  }

  NN_NeuralNetwork nn = nn_alloc(arch.items, arch.count);
  nn_rand(nn, 0.f, 1.f);

  //   const int screenWidth = 800;
  //   const int screenHeight = 600;

  //   InitWindow(screenWidth, screenHeight, "Neural Network Visualizer");
  //   SetTargetFPS(60);

  return 0;
}
