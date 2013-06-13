#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define LIMIT     10000     /* The largest integer we will consider */
#define MASTER     0       /* Rank of master task */

int isprime(int n) {
	int i,squareroot;
	if (n>10) {
		squareroot = (int) sqrt(n);
		for (i=3; i<=squareroot; i=i+2)
			if ((n%i)==0) return 0;
		return 1;
	}
	/* Assume first four primes are counted elsewhere. Forget everything else */
	else
		return 0;
}

int main (int argc, char *argv[]) {
	int ntasks;               /* total number of tasks in partitiion */
	int rank;                 /* task identifier */
	int pcsum;                /* number of primes found by all tasks */
	int maxprime;             /* largest prime found */
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
	if (((ntasks%2) !=0) || ((LIMIT%ntasks) !=0)) {
		printf("An even number of tasks is required.\n");
		printf("The number of tasks shoud divide evenly %d.\n",LIMIT);
		MPI_Finalize();
		exit(0);
	}
	
	double start_time = MPI_Wtime();   /* Initialize start time */
	int mystart = (rank*2)+1;       /* Find my starting point - must be odd number */
	int stride = ntasks*2;          /* Determine stride, skipping even numbers */
	int pc = 0;                     /* Prime counter */
	int foundone = 0;               /* Last prime that was found */
	
	/******************** task with rank MASTER does this part ********************/
	if (rank == MASTER) {
		printf("Using %d tasks to scan %d numbers\n",ntasks,LIMIT);
		int pc = 4; /* We skip the first four primes. */
		for (int n=mystart; n<=LIMIT; n=n+stride) {
			if (isprime(n)) {
				pc++; // found a prime
				foundone = n; // last prime that we have found
				/* Optional: print each prime as it is found */
//				printf("%d\n",foundone);				
			}
		}
			// Total number of primes found by all processes: MPI_SUM
		MPI_Reduce(&pc,&pcsum,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
			// The largest prime that was found by all processes: MPI_MAX
		MPI_Reduce(&foundone,&maxprime,1,MPI_INT,MPI_MAX,MASTER,MPI_COMM_WORLD);
		double end_time=MPI_Wtime();
		printf("Done. Largest prime is %d.\nTotal number of primes found: %d\n",maxprime,pcsum);
		printf("Wallclock time elapsed: %.2lf seconds\n",end_time-start_time);
	}	
	
	/******************** all other tasks do this part ***********************/
	if (rank != MASTER) {
		for (int n=mystart; n<=LIMIT; n=n+stride) {
			if (isprime(n)) {
				pc++;
				foundone = n;
				/* Optional: print each prime as it is found */
//				printf("%d\n",foundone);
			}
		}
		MPI_Reduce(&pc,&pcsum,1,MPI_INT,MPI_SUM,MASTER,MPI_COMM_WORLD);
		MPI_Reduce(&foundone,&maxprime,1,MPI_INT,MPI_MAX,MASTER,MPI_COMM_WORLD);
	}
	
	MPI_Finalize();
}
