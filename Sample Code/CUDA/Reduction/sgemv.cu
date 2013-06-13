#include <iostream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "../utils.h"
#include "../timer.h"

//How to NOT write a matrix vector product
__global__
void sgemvKernel(const int* const A, const int* const x, int* b, int numRows, int numCols)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= numRows)
    return;

  int rowSum = 0;
  for (int c = 0; c < numCols; ++c) {
    rowSum += A[row * numCols + c] * x[c];
  }

  b[row] = rowSum;
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

  const int numThreads = 192;
  const int numBlocks = (numRows + numThreads - 1) / numThreads;
  sgemvKernel<<<numBlocks, numThreads>>>(d_A, d_x, d_b, numRows, numCols);

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
