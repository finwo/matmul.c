SRC=$(wildcard src/*.c)

FIND=$(shell which gfind find | head -1)

INCLUDES?=
INCLUDES+=-I src

CFLAGS?=-Wall -std=c99

include lib/.dep/config.mk

CGLAGS+=$(INCLUDES)

OBJ=$(SRC:.c=.o)

BIN=\
	test_matmul

default: $(BIN)

$(BIN): $(OBJ) test/$(BIN:=.o)
	$(CC) $(CFLAGS) $(OBJ) test/$@.o -o $@

.PHONY: clean
clean:
	rm -f $(OBJ)
	rm -f $(BIN:=.o)

# README.md: ${SRC} src/matmul.h
# 	stddoc < src/matmul.h > README.md

.PHONY: format
format:
	$(FIND) src/ -type f \( -name '*.c' -o -name '*.h' \) -exec clang-format -i {} +
