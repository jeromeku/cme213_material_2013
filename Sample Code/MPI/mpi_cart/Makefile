CC=mpic++
FLAGS=-O2
DEPS =
OBJ = mpi_cart.o
EXE = mpi_cart

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(FLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS)

run:
	mpirun -n 6 ./$(EXE)

clean:
	rm -f $(OBJ) $(EXE)
