CC=mpic++
FLAGS=-O2
DEPS =
OBJ = mpi_group.o
EXE = mpi_group

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(FLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS)

run:
	mpirun -n 8 ./$(EXE)

clean:
	rm -f $(OBJ) $(EXE)
