#include <vector>
#include <iostream>
#include <limits>
#include "../utils.h"

__global__
void reduceWarp(const int* const input, int *sum)
{
  const int lane = threadIdx.x;

  __shared__ int smem[32];

  int myVal = input[lane];

  smem[lane] = myVal;

  __syncthreads();

  for (int shift = 16; shift > 0; shift >>= 1) {
    if (lane < shift) {
      smem[lane] += smem[lane + shift];
    }
    __syncthreads();
  }

  if (lane == 0)
    *sum = smem[lane];
}

int main(void) {
  const int N = 32;

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

  reduceWarp<<<1, 32>>>(d_input, d_sum);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int h_d_sum;
  checkCudaErrors(cudaMemcpy(&h_d_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "cpu: " << h_sum << " gpu: " << h_d_sum << std::endl;


  return 0;
}
