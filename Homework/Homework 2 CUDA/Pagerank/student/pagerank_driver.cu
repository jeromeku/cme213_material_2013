// This is machine problem 1, part 2, page ranking
// The problem is to compute the rank of a set of webpages
// given a link graph, aka a graph where each node is a webpage,
// and each edge is a link from one page to another.
// We're going to use the Pagerank algorithm (http://en.wikipedia.org/wiki/Pagerank),
// specifically the iterative algorithm for calculating the rank of a page
// Implement the corresponding code in CUDA.


#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <ctime>
#include <limits>
#include <vector>

#include "mp1-util.h"

event_pair timer;
typedef unsigned int uint;

#include "studentSolutionPagerank.cu"

// amount of floating point numbers between answer and computed value 
// for the answer to be taken correctly. 2's complement magick.
const int maxUlps = 100;
  
void host_graph_propagate(uint *graph_indices
                        , uint *graph_edges
                        , float *graph_nodes_in
                        , float *graph_nodes_out
                        , float * inv_edges_per_node
                        , int num_nodes)
{
  //for each node
  for(int i=0; i < num_nodes; i++)
  {
    float sum = 0.f; 
    //for all of its edges
    for(int j = graph_indices[i]; j < graph_indices[i+1]; j++)
    {
      sum += graph_nodes_in[ graph_edges[j] ] * inv_edges_per_node[ graph_edges[j] ];
    }
    graph_nodes_out[i] = 0.5f/(float)num_nodes + 0.5f*sum;
  }
}

double host_graph_iterate(uint *graph_indices
                        , uint *graph_edges
                        , float *graph_nodes_in
                        , float *graph_nodes_out
                        , float * inv_edges_per_node
                        , int nr_iterations
                        , int num_nodes)
{
  float *buffer_1 = new float[num_nodes];
  float *buffer_2 = new float[num_nodes];

  memcpy(buffer_1, graph_nodes_in, num_nodes * sizeof(float));

  start_timer(&timer);
  for(int iter = 0; iter < nr_iterations / 2; iter++)
  {
    host_graph_propagate(graph_indices, graph_edges, buffer_1, buffer_2, inv_edges_per_node, num_nodes);
    host_graph_propagate(graph_indices, graph_edges, buffer_2, buffer_1, inv_edges_per_node, num_nodes);
  }

  //handle the odd case and copy memory to the output location
  if (nr_iterations % 2) {
    host_graph_propagate(graph_indices, graph_edges, buffer_1, buffer_2, inv_edges_per_node, num_nodes);
    memcpy(graph_nodes_out, buffer_2, num_nodes * sizeof(float));
  }
  else {
    memcpy(graph_nodes_out, buffer_1, num_nodes * sizeof(float));
  }
  double cpu_elapsed_time = stop_timer(&timer);

  delete [] buffer_1;
  delete [] buffer_2;

  return cpu_elapsed_time;
}

void generateGraph(int num_nodes
                 , int avg_edges
                 , std::vector<uint> &h_graph_indices
                 , std::vector<uint> &h_graph_edges
                 , std::vector<float> &h_inv_edges_per_node
                 , std::vector<float> &h_node_values_input
                 , std::vector<float> &h_gpu_node_values_output
                 , std::vector<float> &h_cpu_node_values_output)
{
  h_graph_indices.     resize(num_nodes + 1);
  h_node_values_input. resize(num_nodes);
  h_inv_edges_per_node.resize(num_nodes);
  h_graph_edges.       resize(num_nodes * avg_edges);

  h_gpu_node_values_output.resize(num_nodes);
  h_cpu_node_values_output.resize(num_nodes);

  h_graph_indices[0] = 0;
  int nodes_per_block = num_nodes / (avg_edges * 2 - 1) + 1;
  for(int i = 0; i < num_nodes; i++)
  {
    //each node has a deterministic number of edges that goes 
    //1, 1, 1, 1, ..., 2, 2, 2, 2, ..., 2 * avg_edges - 1
    int nr_edges = i / nodes_per_block + 1;
    h_inv_edges_per_node[i] = 1.f/(float)nr_edges;
    h_graph_indices[i+1] = h_graph_indices[i] + nr_edges;

    //assign a random node for each edge
    for(int j=h_graph_indices[i];j<h_graph_indices[i+1];j++)
    {
      h_graph_edges[j] = rand() % num_nodes;
    }
    
    h_node_values_input[i] =  1.f/(float)num_nodes; //assign the starting weight
  }
}

void checkErrors(const std::vector<float> &h_gpu_node_values_output, const std::vector<float> &h_cpu_node_values_output) {
  assert(h_gpu_node_values_output.size() == h_cpu_node_values_output.size());
  int num_errors = 0;
  for(int i = 0 ; i < h_gpu_node_values_output.size() ; i++)
  {
    float n = h_gpu_node_values_output[i];
    float c = h_cpu_node_values_output[i];
    if(!AlmostEqual2sComplement(n,c,maxUlps)) 
    {
      num_errors++;
      if (num_errors < 10)
      {
        std::cerr << "\nMismatch at node " << i << std::endl;
        std::cerr << "Expected: " << c << " and got " << n << std::endl;
      }
      else {
        std::cerr << "Too many errors, quitting" << std::endl;
        exit(1);
      }
    }
  }
  if (num_errors) {
    std::cerr << "There were errors, quitting" << std::endl;
    exit(1);
  }
}


int main(void)
{
  //initalize CUDA and warmup kernel to avoid including these costs in the timings
  cudaFree(0);
  device_graph_propagate<<<1, 1>>>(NULL, NULL, NULL, NULL, NULL, 0);

  int iterations = 6;

  std::vector<uint> num_nodes;
  std::vector<uint> avg_edges;

  num_nodes.push_back(1 << 15);
  for (int i = 0; i < 5; ++i) num_nodes.push_back(num_nodes.back() * 2);

  for (int i = 2; i < 20; ++i) avg_edges.push_back(i);
  
  // index array has to be n+1 so that the last thread can 
  // still look at its neighbor for a stopping point
  std::vector<uint> h_graph_indices;
  std::vector<uint> h_graph_edges;
  std::vector<float> h_inv_edges_per_node;
  std::vector<float> h_node_values_input;
  std::vector<float> h_gpu_node_values_output;
  std::vector<float> h_cpu_node_values_output;
  
  // generate random input
  // initialize
  srand(time(NULL));
  
  std::cout << std::setw(15) << " ";
  for (std::vector<uint>::iterator nodes = num_nodes.begin(); nodes != num_nodes.end(); ++nodes) {
    std::cout << std::setw(15) << *nodes;
  }
  std::cout << std::endl << std::endl;
  for (std::vector<uint>::iterator edges = avg_edges.begin(); edges != avg_edges.end(); ++edges) {
    std::cout << std::setw(15) << *edges;
    for (std::vector<uint>::iterator nodes = num_nodes.begin(); nodes != num_nodes.end(); ++nodes) {
      generateGraph(*nodes, *edges, h_graph_indices, h_graph_edges, 
                    h_inv_edges_per_node, h_node_values_input,
                    h_gpu_node_values_output, h_cpu_node_values_output);

      double gpu_time = device_graph_iterate(h_graph_indices.data(), 
                                             h_graph_edges.data(), 
                                             h_node_values_input.data(), 
                                             h_gpu_node_values_output.data(), 
                                             h_inv_edges_per_node.data(), 
                                             iterations, *nodes, *edges);

      // generate reference output
      double cpu_time = host_graph_iterate(h_graph_indices.data(), 
                                           h_graph_edges.data(), 
                                           h_node_values_input.data(), 
                                           h_cpu_node_values_output.data(), 
                                           h_inv_edges_per_node.data(), 
                                           iterations, *nodes);


      // check CUDA output versus reference output
      checkErrors(h_gpu_node_values_output, h_cpu_node_values_output);
      //TODO fill in the calculation for totalBytes
      size_t totalBytes = 0;
      std::cout << std::setw(15) << std::fixed << std::setprecision(2) << totalBytes / (gpu_time / 1000.) / 1E9 << std::flush;
    }
    std::cout << std::endl;
  }
}
