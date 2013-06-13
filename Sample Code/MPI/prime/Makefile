CC=mpic++
FLAGS=-O2
DEPS =
OBJ = mpi_prime.o
EXE = mpi_prime

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(FLAGS)

%.o: %.C $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS)

run:
	mpirun -n 4 ./$(EXE)

clean:
	rm -f $(OBJ) $(EXE)
