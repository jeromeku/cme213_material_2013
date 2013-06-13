#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

using namespace std;

	// Matrix A
float a_val(int i, int j, int n) {
	return 1./(1. + float(abs(i - j))/float(n));
}

	// Vector b
float b_val(int i, int n) {
	return cos(i/n);
}

int main(int argc, char * argv[]) {
	
	const int n = 128; // size of matrix
	
	const int ROW=0, COL=1; /* To improve readability */
	
	MPI_Init(&argc,&argv);	
	
	/* Get information about the communicator */
	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	/* Compute the size of the square grid.
	 We assume that nprocs is a square and that the matrix size
	 is a multiple of sqrt(nprocs). */
	int dims[2];
	dims[ROW] = dims[COL] = sqrt(nprocs);
	assert(dims[ROW] * dims[COL] == nprocs); // Test that nprocs is a square
	int nlocal = n/dims[ROW];
	assert(n % dims[ROW] == 0); // Must divide exactly.
	
	/* Set up the Cartesian topology and get the rank & 
	 coordinates of the process in this topology */
	int periods[2];
	periods[ROW] = periods[COL] = 1; 
	/* We will use wrap-around connections. */
	
	MPI_Comm comm_2d;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);
	
	/* Get my rank in the new topology */
	int my2drank;
	MPI_Comm_rank(comm_2d, &my2drank);
	
	/* Get my coordinates */
	int mycoords[2];	
	MPI_Cart_coords(comm_2d, my2drank, 2, mycoords); 	
	
	/* Create the row-based sub-topology */
	int keep_dims[2];	
	keep_dims[ROW] = 0;
	keep_dims[COL] = 1;
	MPI_Comm comm_row;	
	MPI_Cart_sub(comm_2d, keep_dims, &comm_row);
	
	/* Create the column-based sub-topology */
	keep_dims[ROW] = 1;
	keep_dims[COL] = 0;
	MPI_Comm comm_col;	
	MPI_Cart_sub(comm_2d, keep_dims, &comm_col);
	
	/* Global index offsets */
	const int i_offset = mycoords[ROW]*nlocal;	
	const int j_offset = mycoords[COL]*nlocal;
	
	/* Initialize matrix A */
	vector<float> a(nlocal*nlocal);
	for (int i=0; i<nlocal; ++i) {
		int i_global = i+i_offset; // Global row index		
		for (int j=0; j<n; ++j) {
			int j_global = j+j_offset; // Global column index	
			a[i*nlocal+j] = a_val(i_global,j_global,n);
		}
	}
	
	/* Initialize vector b */
	vector<float> b(nlocal);	
	if (mycoords[COL] == 0) {
		for (int i=0;i<nlocal;++i) {
			int i_global = i+i_offset; // Global row index
			b[i] = b_val(i_global,n);
		}	
	}	
	
	/* Distribute the b vector. */
	/* Step 1. The processes along the 0th column 
	 send their data to the diagonal process. */
	int drank, coords[2];	
		// Send to diagonal block
	if (mycoords[COL] == 0 && mycoords[ROW] != 0) { 
		/* I'm in the first column */
		coords[ROW] = mycoords[ROW];
		coords[COL] = mycoords[ROW]; // coordinates of diagonal block
		MPI_Cart_rank(comm_2d, coords, &drank); // 2D communicator
		/* Send data to the diagonal block */
		MPI_Send(&b[0], nlocal, MPI_FLOAT, drank, 1, comm_2d);
	}
	
	int col0rank;		
	coords[ROW] = mycoords[ROW];
	coords[COL] = 0; // Receiving from column 0
	MPI_Cart_rank(comm_2d, coords, &col0rank); // 2D communicator
	
		// Receive from column 0
	if (mycoords[ROW] == mycoords[COL] && mycoords[ROW] != 0) {
		/* I am a diagonal block */
		MPI_Recv(&b[0], nlocal, MPI_FLOAT, col0rank, 1, comm_2d,
				 MPI_STATUS_IGNORE);
	}	
	
	/* Step 2. The diagonal processes perform a 
	 column-wise broadcast */
	coords[0] = mycoords[COL]; 
	/* Column sub-topology */
	MPI_Cart_rank(comm_col, coords, &drank);
	MPI_Bcast(&b[0], nlocal, MPI_FLOAT, drank, comm_col);
	
	/* Get into the main computational loop: A*b */
	vector<float> px(nlocal);	
	for (int i=0; i<nlocal; i++) {
		px[i] = 0.0;
		for (int j=0; j<nlocal; j++)
			px[i] += a[i*nlocal+j]*b[j];
	}
	
	/* Perform the sum-reduction along the rows to add up 
	 the partial dot-products; result is stored in column 0. */
	coords[0] = 0;
	/* Row sub-topology */
	MPI_Cart_rank(comm_row, coords, &col0rank);
	vector<float> x(nlocal);	
	MPI_Reduce(&px[0], &x[0], nlocal, MPI_FLOAT, MPI_SUM, col0rank, comm_row);
	
	/* Test */
	if (mycoords[COL] == 0) {
		float emax = 0.;
		for (int i=0; i<nlocal; i++) {
			float x0 = 0.0;
			int i_global = i+i_offset; // Global row index
			for (int j=0; j<n; j++) {			
				float a0 = a_val(i_global,j,n);
				float b0 = b_val(j,n);
				x0 += a0*b0;
			}
			emax = max(emax,abs(x0-x[i]));
		}			
		printf("Row block = %d, error = %9.3e\n",mycoords[ROW],emax);		
	}
	
	MPI_Finalize();	
}