CFLAGS+=-mfma -mavx2 -mavxvnni -mavx512f -mavx512vnni -mavx512bw -fopenmp

SRC+={{module.dirname}}/src/matmul.c
