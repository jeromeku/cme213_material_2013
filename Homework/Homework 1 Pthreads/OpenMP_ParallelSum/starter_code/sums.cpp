#include<vector>
#include<iostream>
#include<omp.h>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include "tests.h"

#define MAX_INT 100
#define SIZE 30000000
typedef unsigned int uint; 

std::vector<uint> serialSum(std::vector<uint> &v) {
    std::vector<uint> sums(2);
    // TO DO
    return sums;
}
std::vector<uint> parallelSum(std::vector<uint> &v) {
    std::vector<uint> sums(2);
    // TO DO
    return sums;
}

std::vector<uint> initializeRandomly(uint size, uint max_int) { 
    std::vector<uint> res(size);
    for (uint i = 0; i < size; ++i) { 
        res[i] = rand() % max_int;
    }
    return res;
}

int main(int argc, char** argv){
   
    // you can uncomment the line below to make your own simple tests
    // std::vector<uint> v = ReadVectorFromFile("vec");
    
    std::vector<uint> v = initializeRandomly(SIZE, MAX_INT);
    std::cout << "Parallel" << std::endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    std::vector<uint> sums = parallelSum(v);
    std::cout << sums[0] << std::endl;
    std::cout << sums[1] << std::endl;
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    std::cout << delta << std::endl;

    std::cout << "Serial" << std::endl;
    gettimeofday(&start, NULL);
    std::vector<uint> sumsSer = serialSum(v);
    std::cout << sumsSer[0] << std::endl;
    std::cout << sumsSer[1] << std::endl;
    gettimeofday(&end, NULL);
    delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    std::cout << delta << std::endl;

    return 0;
}

