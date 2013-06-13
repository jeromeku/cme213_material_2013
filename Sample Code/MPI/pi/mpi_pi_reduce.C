#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>

void srandom (unsigned seed);

float dboard (int darts);

#define DARTS 10000    /* number of throws at dartboard */
#define ROUNDS 10      /* number of times "darts" is iterated */
#define MASTER 0       /* task ID of master task */

int main (int argc, char *argv[]) {
	int	taskid,	    /* task ID */
	    numtasks;   /* number of tasks */
	
	/* Obtain number of tasks and task ID */
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	printf ("MPI task %d has started...\n", taskid);
	
	/* Seed random number generator */
	int seed = taskid + time(NULL);
	srand(seed);
	
	float avepi = 0.;
	long int throws_total = 0;
	
	for (int i = 0; i < ROUNDS; i++) {
		
		int nthrows = DARTS*(1<<i);	 // Number of dart throws	
		throws_total += nthrows*numtasks;
		
		/* All processes calculate pi using dartboard algorithm */		
		float homepi = dboard(nthrows);
		float pisum;
		
		/* Retrieve the sum of all values computed by each process */		
		int rc = MPI_Reduce(&homepi, &pisum, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		if (rc != MPI_SUCCESS)
			printf("%d: failure on mpi_reduce\n", taskid);
		
		/* Master computes average for this iteration and all iterations */
		if (taskid == MASTER) {
			avepi += pisum/numtasks;
			printf("   After %10d throws, average value of pi = %10.8f\n",
				   throws_total,avepi/(i+1));
		}    
	} 
	
	if (taskid == MASTER) 
		printf ("\nReal value of PI: 3.1415926535897 \n");
	
	MPI_Finalize();
	return 0;
}