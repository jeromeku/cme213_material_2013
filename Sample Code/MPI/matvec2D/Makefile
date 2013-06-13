CC=mpic++
FLAGS=-O2
DEPS =
OBJ = matvec2d.o
EXE = matvec2d

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(FLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(FLAGS)

run:
	mpirun -n 16 ./$(EXE)

clean:
	rm -f $(OBJ) $(EXE)
