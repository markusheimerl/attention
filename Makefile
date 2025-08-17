CC = clang
CFLAGS = -O3 -march=native -fno-math-errno -funsafe-math-optimizations -fno-rounding-math -Wall -Wextra
LDFLAGS = -lm -flto

train.out: data.o train.o
	$(CC) data.o train.o $(LDFLAGS) -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin

.PHONY: run clean