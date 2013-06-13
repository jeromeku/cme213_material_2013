#include "mpi.h"
#include <cstdio>
#include <cstdlib>

#define NPROCS 8

int main(int argc, char * argv[]) {
	int rank, nprocs;	
	int ranks1[4]={0,1,2,3}, ranks2[4]={4,5,6,7};
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	
	if (nprocs != NPROCS) {
		printf("The number of processes must be %d. Terminating.\n",NPROCS);
		MPI_Finalize();
		exit(0);
	}
	
	/* Extract the original group handle */
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	
	/* Divide tasks into two distinct groups based upon rank */
	MPI_Group sub_group;	
	if (rank < NPROCS/2) {
		MPI_Group_incl(world_group, NPROCS/2, ranks1, &sub_group);
	}
	else {
		MPI_Group_incl(world_group, NPROCS/2, ranks2, &sub_group);
	}
	
	/* Create new new communicator and then perform collective communications */
	MPI_Comm sub_group_comm;	
	MPI_Comm_create(MPI_COMM_WORLD, sub_group, &sub_group_comm);
	int sendbuf = rank;	
	int recvbuf;
	MPI_Allreduce(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, sub_group_comm);
	
	int group_rank;	
	MPI_Group_rank(sub_group, &group_rank);
	printf("rank= %d group rank= %d recvbuf= %d\n",rank,group_rank,recvbuf);
	
	MPI_Finalize();
}