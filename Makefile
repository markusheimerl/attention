CC = clang
CFLAGS = -O3 -march=native -fno-math-errno -funsafe-math-optimizations -fno-rounding-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

train.out: attention.o data.o train.o
	$(CC) attention.o data.o train.o $(LDFLAGS) -o $@

attention.o: attention.c attention.h
	$(CC) $(CFLAGS) -c attention.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c attention.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) clean -C gpu