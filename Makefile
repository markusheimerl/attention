CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
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
	$(MAKE) -C gpu clean

test_buffer_sharing.out: attention.o test_buffer_sharing.o
	$(CC) attention.o test_buffer_sharing.o $(LDFLAGS) -o $@

test_buffer_sharing.o: test_buffer_sharing.c attention.h
	$(CC) $(CFLAGS) -c test_buffer_sharing.c -o $@

test: test_buffer_sharing.out
	./test_buffer_sharing.out