#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
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

template<int blockSize>
__global__
void reduceMultiBlock(const int* const input, int *sum, int N)
{
  const int tid  = threadIdx.x;
  const int gtid = blockIdx.x * blockSize + threadIdx.x;

  __shared__ int smem[blockSize];

  int myVal = 0;

  //first do a serial accumulation into each thread's local accumulator
  for (int globalPos = gtid; globalPos < N; globalPos += blockSize * gridDim.x) {
    myVal += input[globalPos];
  }

  int blockSum = blockReduce<blockSize>(myVal, smem);

  if (tid == 0)
    atomicAdd(sum, blockSum);
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);

  std::vector<int> h_input(N);

  int h_sum = 0.f;

  for (int i = 0; i < N; ++i) {
    h_input[i] = rand() % 10;
    h_sum += h_input[i];
  }

  int *d_input;
  checkCudaErrors(cudaMalloc(&d_input, N * sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_input, &h_input[0], N * sizeof(int), cudaMemcpyHostToDevice));

  int *d_sum;
  checkCudaErrors(cudaMalloc(&d_sum, sizeof(int)));
  checkCudaErrors(cudaMemset(d_sum, 0, sizeof(int)));

  const int blockSize = 192;
  reduceMultiBlock<blockSize><<<1, 1>>>(d_input, d_sum, N); //warm up kernel for timing

  std::cout << std::setw(12) << "Block Size" << " " << std::setw(12) << "Bandwidth" << std::endl;
  for (int maxBlocks = 32; maxBlocks <= 256; maxBlocks += 2) {
    const int numBlocks = min( ((N+1)/2 + blockSize - 1) / blockSize, maxBlocks);

    GpuTimer timer; timer.Start();

    reduceMultiBlock<blockSize><<<numBlocks, blockSize>>>(d_input, d_sum, N);

    timer.Stop();

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemset(d_sum, 0, sizeof(int)));

    std::cout << std::setw(12) << numBlocks << " " << std::setw(12) << (N * sizeof(int) / 1E6) / timer.Elapsed() << std::endl;
  }

  return 0;
}
