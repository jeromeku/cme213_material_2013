CC=mpic++
FLAGS=-O2
DEPS =
OBJ = matvecrow.o
EXE = matvecrow

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(FLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS)

run:
	mpirun -n 4 ./$(EXE)

clean:
	rm -f $(OBJ) $(EXE)
