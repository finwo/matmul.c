SRC=$(wildcard src/*.c)

FIND=$(shell which gfind find | head -1)

INCLUDES?=
INCLUDES+=-I src

CFLAGS?=-Wall -std=c99

include lib/.dep/config.mk

CFLAGS+=-D_DEFAULT_SOURCE -mfma -mavx2 -mavxvnni -mavx512f -mavx512vnni -mavx512bw -fopenmp -O3

CFLAGS+=$(INCLUDES)

OBJ=$(SRC:.c=.o)

BIN=\
	test_matmul \
	benchmark

default: $(BIN)

test/%.o: test/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test_matmul: $(OBJ) test/test_matmul.o
	$(CC) $(CFLAGS) $(OBJ) test/test_matmul.o -o $@ -lrt -fopenmp

benchmark: $(OBJ) test/benchmark.o
	$(CC) $(CFLAGS) $(OBJ) test/benchmark.o -o $@ -lrt -fopenmp

.PHONY: clean
clean:
	rm -f $(OBJ)
	rm -f $(BIN:=.o)

# README.md: ${SRC} src/matmul.h
# 	stddoc < src/matmul.h > README.md

.PHONY: format
format:
	$(FIND) src/ -type f \( -name '*.c' -o -name '*.h' \) -exec clang-format -i {} +
	$(FIND) test/ -type f \( -name '*.c' -o -name '*.h' \) -exec clang-format -i {} +
