CC = clang
CFLAGS = -Wall -Werror -O3
LIBS = -lm -ldl 
BUILD_DIR = build

TARGETS = ml_helloWorld ml_withBoolean ml_xor mat_xor nn_xor nn_test nn_fa nn_viz
BINARIES = $(addprefix $(BUILD_DIR)/, $(TARGETS))

# Raylib configuration (macOS specific with pkg-config)
RAYLIB_CFLAGS = $(shell pkg-config --cflags raylib)
RAYLIB_LIBS = $(shell pkg-config --libs raylib) -pthread

all: $(BUILD_DIR) $(BINARIES)

.PHONY: all clean format

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule for simple examples in basics/
$(BUILD_DIR)/ml_%: basics/ml_%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Rule for the main nn library example
$(BUILD_DIR)/mat_xor: nn/examples/mat_xor.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Rule for the nn library test suite
$(BUILD_DIR)/nn_test: nn/examples/nn_test.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Rule for the nn library test suite
$(BUILD_DIR)/nn_xor: nn/examples/nn_xor.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/nn_fa: nn/examples/nn_fa.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(BUILD_DIR)/nn_viz: nn/nn_viz.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RAYLIB_CFLAGS) -o $@ $< $(LIBS) $(RAYLIB_LIBS)

clean:
	rm -rf $(BUILD_DIR)

format:
	clang-format --style=llvm -i basics/*.c nn/examples/*.c nn/*.c nn/*.h
