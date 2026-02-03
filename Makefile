CC = clang
CFLAGS = -Wall -Werror -O3
LIBS = -lm
BUILD_DIR = build

TARGETS = ml_helloWorld ml_withBoolean ml_xor mat_xor nn_xor nn_test 
BINARIES = $(addprefix $(BUILD_DIR)/, $(TARGETS))

.PHONY: all clean format

all: $(BUILD_DIR) $(BINARIES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule for simple examples in basics/
$(BUILD_DIR)/ml_%: basics/ml_%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Rule for the main nn library example
$(BUILD_DIR)/mat_xor: nn/mat_xor.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Rule for the nn library test suite
$(BUILD_DIR)/nn_test: nn/nn_test.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Rule for the nn library test suite
$(BUILD_DIR)/nn_xor: nn/nn_xor.c nn/nn.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf $(BUILD_DIR)

format:
	clang-format --style=llvm -i basics/*.c nn/*.c nn/*.h
