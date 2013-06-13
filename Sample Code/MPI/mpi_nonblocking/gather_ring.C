#include <mpi.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
		// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	
		// Find out the rank and size
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int nproc;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
	srand(rank); // Initialize the random number generator
	int number_send, number_recv;
	number_send = rand() % 100;
	
	const int rank_receiver = rank==nproc-1 ? 0 : rank + 1;
	const int rank_sender   = rank==0 ? nproc-1 : rank-1;
	
		// Size of array needed to store result
	int * numbers = new int[nproc];
	for (int i=0; i<nproc; ++i) numbers[rank] = -1;
	
	numbers[rank] = number_send;
	printf("Number for process %d: %2d\n",rank,numbers[rank]);
	
	for (int i=0;i<nproc-1;++i) {
			// Send to the right: Isend
		int * p_send = numbers + (rank - i + nproc) % nproc;
		MPI_Request send_req;
		MPI_Isend(p_send, 1, MPI_INT, rank_receiver, 0, MPI_COMM_WORLD, &send_req);
		
			// Receive from the left: Irecv
		int * p_recv = numbers + (rank - i - 1 + nproc) % nproc;		
		MPI_Request recv_req;		
		MPI_Irecv(p_recv, 1, MPI_INT, rank_sender, 0, MPI_COMM_WORLD, &recv_req);
			// You can try a Recv as well! In that case, comment out the MPI_Wait() below
			// on line 49.
			// MPI_Recv(p_recv, 1, MPI_INT, rank_sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
		
			// Wait for communication to complete
		MPI_Status status;
			// This wait is actually not required for correctness
		MPI_Wait(&send_req, &status); // Isend
		
			// This wait is required with Irecv
		MPI_Wait(&recv_req, &status); // Irecv
	}
	
	if (rank == 0) {
		for (int i=0;i<nproc;++i) 
			printf("Numbers gathered at root node: %d %2d\n",i,numbers[i]);
	}
	
	delete [] numbers;
	
	MPI_Finalize();
}
