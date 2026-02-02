all: ml_helloWorld ml_withBoolean ml_xor nn

ml_helloWorld: basics/ml_helloWorld.c
		clang -o build/ml_helloWorld -Wall -Werror basics/ml_helloWorld.c

ml_withBoolean: basics/ml_withBoolean.c
		clang -o build/ml_withBoolean -Wall -Werror -lm basics/ml_withBoolean.c

ml_xor: basics/ml_xor.c
		clang -o build/ml_xor -Wall -Werror -lm basics/ml_xor.c

nn: nn/nn.c nn/nn.h
		clang -o build/nn -Wall -Werror -lm nn/nn.c

format: basics/ml_helloWorld.c basics/ml_withBoolean.c basics/ml_xor.c nn/nn.h nn/nn.c
		clang-format --style=llvm -i basics/ml_helloWorld.c
		clang-format --style=llvm -i basics/ml_withBoolean.c
		clang-format --style=llvm -i basics/ml_xor.c
		clang-format --style=llvm -i nn/nn.h
		clang-format --style=llvm -i nn/nn.c
