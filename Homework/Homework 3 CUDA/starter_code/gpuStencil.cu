#include "BC.h"

template<int order> 
__device__
float Stencil(float *prev, int width, float xcfl, float ycfl) { 
  if (order == 2) {
    return prev[0] + xcfl * (prev[-1] + prev[1] - 2.f * prev[0]) + ycfl * (prev[width] + prev[-width] - 2.f * prev[0]);
  }
  else if (order == 4) { 
    return prev[0] + xcfl * (- prev[2] + 16.f * prev[1] - 30.f * prev[0] + 16.f * prev[-1] - prev[-2]) + 
    ycfl * (- prev[2 * width] + 16.f * prev[width] - 30.f * prev[0] + 16.f * prev[-width] - prev[-2 * width]);
  }
  else { 
    return prev[0] + xcfl * (-9.f * prev[4] + 128.f * prev[3] - 1008.f * prev[2] + 8064.f * prev[1] - 
    14350.f * prev[0] + 8064.f * prev[-1] - 1008.f * prev[-2] + 128.f * prev[-3] - 9.f * prev[-4]) +
    ycfl * (-9.f * prev[4 * width] + 128.f * prev[3 * width] - 1008.f * prev[2 * width] + 8064.f * prev[width] -
    14350.f * prev[0] + 8064.f * prev[-width] - 1008.f * prev[-2 * width] + 128.f * prev[-3 * width] - 9.f * prev[-4 * width]);
  }
}

// Simplest algorithm using global memory

template<int order>
__global__
void gpuStencil(float *curr, float *prev, int gx, int nx, int ny, float xcfl, float ycfl, int borderSize)
{
  // TODO
}

double gpuComputation(Grid &curr_grid, const simParams &params) {

  boundary_conditions BC(params);

  Grid prev_grid(curr_grid);
  // TODO
  dim3 threads(0, 0);
  dim3 blocks(0, 0);

  event_pair timer;
  start_timer(&timer);
  for (int i = 0; i < params.iters(); ++i) {
    curr_grid.swap(curr_grid, prev_grid);

    // update the values on the boundary only
    BC.updateBC(curr_grid.dGrid_, prev_grid.dGrid_);

    // apply stencil    
    // TODO
    
    check_launch("gpuStencil");
  }

  return stop_timer(&timer);
}


// Global memory algorithm with loop for sub-domain

template<int order, int numYPerStep>
__global__
void gpuStencilLoop(float *curr, float *prev, int gx, int nx, int ny, float xcfl, float ycfl, int borderSize)
{
  // TODO  
}

double gpuComputationLoop(Grid &curr_grid, const simParams &params) {

  boundary_conditions BC(params);

  Grid prev_grid(curr_grid);
  // TODO
  dim3 threads(0, 0);
  dim3 blocks(0, 0);

  event_pair timer;
  start_timer(&timer);

  for (int i = 0; i < params.iters(); ++i) {
    curr_grid.swap(curr_grid, prev_grid);

    // update the values on the boundary only
    BC.updateBC(curr_grid.dGrid_, prev_grid.dGrid_);

    // apply stencil  
    // TODO    

    check_launch("gpuStencilLoop");
  }

  return stop_timer(&timer);
}



// Shared memory algorithm

template<int side, int usefulSide, int borderSize, int order>
  __global__
void gpuShared(float *curr, float *prev, int gx, int gy, float xcfl, float ycfl)
{
  // TODO 
}

template<int order>
double gpuComputationShared(Grid &curr_grid, const simParams &params) {

  boundary_conditions BC(params);

  Grid prev_grid(curr_grid);
 
  // TODO
  dim3 threads(0, 0);
  dim3 blocks(0, 0);

  event_pair timer;
  start_timer(&timer);

  for (int i = 0; i < params.iters(); ++i) {
    curr_grid.swap(curr_grid, prev_grid);

    // update the values on the boundary only
    BC.updateBC(curr_grid.dGrid_, prev_grid.dGrid_);

    // apply stencil   
    // TODO    

    check_launch("gpuShared");
  }

  return stop_timer(&timer);
}
