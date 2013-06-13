#include <iostream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "../utils.h"
#include "../timer.h"

__device__
int nextPowerOf2(const int x) {
  return (1 << (32 - __clz(x - 1)));
}

template<int blockSize>
__device__
int blockReduce(const int threadVal, int *smem) {
  const int tid = threadIdx.x;
  smem[tid] = threadVal;

  __syncthreads();

  //use this for non-power of 2 blockSizes
  for (int shift = nextPowerOf2(blockSize) / 2; shift > 0; shift >>= 1) {
    if (tid < shift && tid + shift < blockSize) {
      smem[tid] += smem[tid + shift];
    }
    __syncthreads();
  }

  return smem[0];
}

//This version is fairly optimal for matrices with "enough" rows (> 224) and many columns
template<int blockSize>
__global__
void sgemvKernel(const int* const A, const int* const x, int* b, int numRows, int numCols)
{
  int row = blockIdx.x;

  int tSum = 0;
  for (int c = threadIdx.x; c < numCols; c += blockSize) {
    tSum += A[row * numCols + c] * x[c];
  }

  __shared__ int smem[blockSize];

  int rowSum = blockReduce<blockSize>(tSum, smem);
  if (threadIdx.x == 0) {
    b[row] = rowSum;
  }
}

int main(int argc, char **argv) {
  int numRows, numCols;
  if (argc == 1) {
    numRows = 1024; numCols = 1024;
  }
  else if (argc == 2) {
    numRows = numCols = atoi(argv[1]);
  }
  else {
    numRows = atoi(argv[1]);
    numCols = atoi(argv[2]);
  }

  std::vector<int> h_A(numRows * numCols);
  std::vector<int> h_x(numCols);
  std::vector<int> h_b(numRows);

  for (int i = 0; i < numRows * numCols; ++i)
    h_A[i] = rand() % 10;

  for (int i = 0; i < numCols; ++i)
    h_x[i] = rand() % 10;

  for (int r = 0; r < numRows; ++r) {
    int sum = 0;
    for (int c = 0; c < numCols; ++c) {
      sum += h_A[r * numCols + c] * h_x[c];
    }
    h_b[r] = sum;
  }

  int *d_A, *d_x, *d_b;
  checkCudaErrors(cudaMalloc(&d_A, sizeof(int) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_x, sizeof(int) * numCols));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(int) * numRows));

  checkCudaErrors(cudaMemcpy(d_A, &h_A[0], sizeof(int) * numRows * numCols, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x, &h_x[0], sizeof(int) * numCols, cudaMemcpyHostToDevice));

  GpuTimer timer; timer.Start();

  const int blockSize = 192;
  const int numBlocks = numRows;
  sgemvKernel<blockSize><<<numBlocks, blockSize>>>(d_A, d_x, d_b, numRows, numCols);

  timer.Stop();

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  std::vector<int> h_d_b(numRows);
  checkCudaErrors(cudaMemcpy(&h_d_b[0], d_b, sizeof(int) * numRows, cudaMemcpyDeviceToHost));

  for (int r = 0; r < numRows; ++r) {
    if (h_b[r] != h_d_b[r]) {
      printf("Mismatch at pos %d: %d %d\n", r, h_b[r], h_d_b[r]);
    }
  }

  std::cout << "Took: " << timer.Elapsed() << " ms" << std::endl;
  std::cout << "Bandwidth: " << (2 * numRows * numCols + numCols) * sizeof(int) / timer.Elapsed() / 1E6 << std::endl;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_b));
}
