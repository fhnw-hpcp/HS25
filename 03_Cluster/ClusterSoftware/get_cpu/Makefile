PROGNAME := get_cpu

main: getcpuid.h
	mpicc -std=c99 -fopenmp main.c getcpuid.c -o $(PROGNAME)

clean:
	rm -f $(PROGNAME) *.o
