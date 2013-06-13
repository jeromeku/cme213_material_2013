#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

#define MASTER 0 /* Rank of master task */

using namespace std;

	// Matrix A
float a_val(int i, int j, int n) {
	return 1./(1. + float(abs(i - j))/float(n));
}

	// Vector b
float b_val(int i, int n) {
	return cos(i/n);
}

int main (int argc, char *argv[]) {
	
	const int n = 128; // Size of matrix
	
	MPI_Init(&argc,&argv);	
	
	/* Get information about the communicator */
	int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	assert(n % nprocs == 0); // It must divide exactly.
	
	int nlocal = n/nprocs; 
		// Number of rows this process is going to calculate	
	
	int i_offset = myrank*nlocal; // Global offset for row index
	
	vector<float> a(nlocal*n);
	vector<float> bloc(nlocal);	
	
		// Fill up a and b with some data
	for (int i=0;i<nlocal;++i) {
		int i_global = i+i_offset; // Global row index
		bloc[i] = b_val(i_global,n);
		for (int j=0; j<n; ++j)
			a[i*n+j] = a_val(i_global,j,n);
	}
	
	/* Allocate the memory used to store the entire b */
	vector<float> b(n);
	
	/* Gather entire vector b on each processor using Allgather */
	MPI_Allgather(&bloc[0], nlocal, MPI_FLOAT, &b[0], nlocal, MPI_FLOAT, MPI_COMM_WORLD);
		// sending nlocal and receiving nlocal from any other process
	
	/* Perform the matrix-vector multiplication involving the 
	 locally stored submatrix. */
	vector<float> x(nlocal);	
	for (int i=0; i<nlocal; i++) {
		x[i] = 0.0;
		for (int j=0; j<n; j++)
			x[i] += a[i*n+j]*b[j];
	}
	/* Done! */
	
	/* Test */
	for (int i=0; i<nlocal; i++) {
		float x0 = 0.0;
		int i_global = i+i_offset; // Global row index
		for (int j=0; j<n; j++) {
			float a0 = a_val(i_global,j,n);
			float b0 = b_val(j,n);
			x0 += a0*b0;
		}
		assert(x0 == x[i]);
	}	
	
	MPI_Finalize();
}