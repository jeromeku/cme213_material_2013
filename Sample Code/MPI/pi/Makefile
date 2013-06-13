CC=mpic++
FLAGS=-O2
DEPS =
OBJ = dboard.o mpi_pi_reduce.o
EXE = mpi_pi_reduce

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(FLAGS)

%.o: %.C $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS)

run:
	mpirun -n 4 ./$(EXE)

clean:
	rm -f $(OBJ) $(EXE)
