#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

int main(int argc, char * argv[]) {
	
	MPI_Init(&argc,&argv);
	
	/* Get information about the communicator */
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 
	
	MPI_Comm comm_cart;
	int ndims, dims[2];
	int periods[2], reorder;
	
	ndims = 2; dims[0] = 3; dims[1] = 2;
	periods[0] = 1; periods[1] = 1; reorder = 1;
	
	assert(nprocs >= dims[0]*dims[1]);
	
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, 
					periods, reorder, &comm_cart);
	
	/* Get my rank in the new topology */
	int my2drank;
	MPI_Comm_rank(comm_cart, &my2drank);
	
	/* Get my coordinates */
	int mycoords[2];	
	MPI_Cart_coords(comm_cart, my2drank, 2, mycoords);	
	
	/* Get coordinates of process to my right and up */
	int rank_right,coords[2];	
	coords[0] = mycoords[0]+1;
	coords[1] = mycoords[1];
	MPI_Cart_rank(comm_cart, coords, &rank_right); // Rank to the right
	
	int rank_up;
	coords[0] = mycoords[0];
	coords[1] = mycoords[1]+1;
	MPI_Cart_rank(comm_cart, coords, &rank_up); // Rank up
	
	printf("Process rank=%d coords=(%d,%d); rank right %d rank up %d\n",
		   my2drank,mycoords[0],mycoords[1],rank_right,rank_up);
	
	MPI_Finalize();	
}	
