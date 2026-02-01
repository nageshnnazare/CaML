all: ml_helloWorld ml_withBoolean ml_xor

ml_helloWorld: ml_helloWorld.c
		clang -o build/ml_helloWorld -Wall -Werror ml_helloWorld.c

ml_withBoolean: ml_withBoolean.c
		clang -o build/ml_withBoolean -Wall -Werror -lm ml_withBoolean.c

ml_xor: ml_xor.c
		clang -o build/ml_xor -Wall -Werror -lm ml_xor.c

format: ml_helloWorld.c ml_withBoolean.c
		clang-format --style=llvm -i ml_helloWorld.c
		clang-format --style=llvm -i ml_withBoolean.c
		clang-format --style=llvm -i ml_xor.c
