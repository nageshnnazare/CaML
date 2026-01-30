all:
		make ml_helloWorld
		make ml_withBoolean

ml_helloWorld: ml_helloWorld.c
		gcc -o build/ml_helloWorld -Wall -Werror ml_helloWorld.c

ml_withBoolean: ml_withBoolean.c
		gcc -o build/ml_withBoolean -Wall -Werror ml_withBoolean.c

