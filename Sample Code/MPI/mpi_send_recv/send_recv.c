#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
		// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	
	int world_rank;
		// What is the process ID? It's called rank.
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
		// How many processes total do we have?
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
		// We need at least 2 processes for this task
	if (world_size < 2) {
		fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
			// Use this command to terminate MPI
		MPI_Abort(MPI_COMM_WORLD, 1); 
	}
	
	int number; 
	srand(time(NULL)); // Initialize the random number generator
	
	if (world_rank == 0) {
			// If we are rank 0, send a random number to process 1
		number = rand();
		MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		printf("Process 0 sent number %d to process 1\n", number);    
	} else if (world_rank == 1) {
			// Receive the number from process 0
		MPI_Status status;
		MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		printf("Process 1 received number %d from process 0\n", number);
	}
	
		// Always call at the end
	MPI_Finalize();
}
