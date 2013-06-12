__global__
void device_graph_propagate(uint *graph_indices
                          , uint *graph_edges
                          , float *graph_nodes_in
                          , float *graph_nodes_out
                          , float *inv_edges_per_node
                          , int num_nodes)
{
  //fill in the kernel code here
}

double device_graph_iterate(uint *h_graph_indices
                          , uint *h_graph_edges
                          , float *h_node_values_input
                          , float *h_gpu_node_values_output
                          , float *h_inv_edges_per_node
                          , int nr_iterations
                          , int num_nodes
                          , int avg_edges)
{
  //allocate GPU memory

  //check for allocation failure

  //copy data to the GPU

  start_timer(&timer);

  const int block_size = 192;
  
  //launch your kernels the appropriate number of iterations
  
  check_launch("gpu graph propagate");
  double gpu_elapsed_time = stop_timer(&timer);

  //copy final data back to the host for correctness checking

  //free the memory you allocated!

  return gpu_elapsed_time;
}
