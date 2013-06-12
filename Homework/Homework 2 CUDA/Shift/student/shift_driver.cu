/* This is machine problem 1, part 1, shift problem
 *
 * The problem is to take in a string (a vector of characters) and a shift amount,
 * and add that number to each element of
 * the string, effectively "shifting" each element in the 
 * string.
 * 
 * We do this in three different ways:
 * 1. With a cuda kernel loading chars and outputting chars for each thread
 * 2. With a cuda kernel, casting the character pointer to an int so that
 *    we load and store 4 bytes each time instead of 1 which gives us better coalescing
 *    and uses the memory effectively to achieve higher bandwidth
 * 3. Same spiel except with a uint2, so that we load 8 bytes each time
 *
 */


#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <vector>

#include "mp1-util.h"

#include "studentSolutionShift.cu"

void host_shift(std::vector<unsigned char> &input_array
                     , std::vector<unsigned char> &output_array
                     , unsigned char shift_amount)
{
  for(unsigned int i=0;i<input_array.size();i++)
  {
    unsigned char element = input_array[i];
    output_array[i] = element + shift_amount;
  }
}

void checkResults(std::vector<unsigned char> &text_host
                , unsigned char *device_output_array
                , int num_entries
                , const char *type) 
{
  //allocate space on host for gpu results
  std::vector<unsigned char> text_from_gpu(num_entries);

  // download and inspect the result on the host:
  cudaMemcpy(&text_from_gpu[0], device_output_array, num_entries, cudaMemcpyDeviceToHost);
  check_launch("copy from gpu");

  // check CUDA output versus reference output
  int error = 0;
  for(int i = 0; i < num_entries; i++)
  {
    if(text_host[i] != text_from_gpu[i]) 
    { 
      ++error;
      std::cerr << "Error at pos: " << i << "\nexpected: " << (int)text_host[i] << 
        " got: " << (int)text_from_gpu[i] << std::endl;
      if (error > 20)
        break;
    }
  }

  if(error)
  {
    std::cerr << "\nError(s) in " << type << " kernel!" << std::endl;
    exit(1);
  }
}

int main(int argc, char ** argv)
{
  //check that the correct number of commandline arguments were given
  if (argc != 2) {
    std::cerr << "Must supply the number of times to double the input file!" << std::endl;
    return 1;
  }

  int number_of_doubles = atoi(argv[1]); //convert argument to integer

  cudaFree(0); //initialize cuda context to avoid including cost in timings later

  //warm-up each of the kernels to avoid including overhead in timing
  //if the kernels are written correctly then they should
  //never make a bad memory access, even though we are passing in NULL
  //pointers since we are also passing in a size of 0
  shift_char <<<1, 1>>>(NULL, NULL, 0, 0);
  shift_int  <<<1, 1>>>(NULL, NULL, 0, 0);
  shift_int2 <<<1, 1>>>(NULL, NULL, 0, 0);

  //First load the text 
  std::string input_file("mobydick.txt");
  std::ifstream ifs(input_file.c_str(), std::ios::binary);
  if (!ifs.good()) {
      std::cerr << "Couldn't open " << input_file << "!" << std::endl;
      return 1;
  }

  std::vector<unsigned char> text;

  ifs.seekg(0, std::ios::end); //seek to end of file
  int length = ifs.tellg();    //get distance from beginning
  ifs.seekg(0, std::ios::beg); //move back to beginning

  text.resize(length);
  ifs.read((char *)&text[0], length);

  ifs.close();

  //need to make a couple copies of the book, otherwise everything happens too quickly
  //make 2^4 = 16 copies
  std::vector<uint> sizes_to_test;
  sizes_to_test.push_back(text.size());

  for (int i = 0; i < number_of_doubles; ++i) {
      text.insert(text.end(), text.begin(), text.end());
      sizes_to_test.push_back(text.size());
  }

  // compute the size of the arrays in bytes
  // with enough padding that a uint2 access won't be out of bounds
  int num_bytes = (text.size() + 7) * sizeof(unsigned char);

  //allocate host arrays
  std::vector<unsigned char> text_gpu(text.size());
  std::vector<unsigned char> text_host(text.size());

  // pointers to device arrays
  unsigned char *device_input_array  = 0;
  unsigned char *device_output_array = 0;
  
  // cudaMalloc device arrays
  cudaMalloc((void**)&device_input_array,  num_bytes);
  cudaMalloc((void**)&device_output_array, num_bytes);
  
  // if either memory allocation failed, report an error message
  if(device_input_array == 0 || device_output_array == 0)
  {
    std::cerr << "Couldn't allocate memory!" << std::endl;
    return 1;
  }

  // generate random shift
  unsigned char shift_amount = (rand() % 25) + 1; //we don't want the shift to be 0!

  // copy input to GPU
  {
    event_pair timer;
    start_timer(&timer);
    cudaMemcpy(device_input_array, &text[0], num_bytes, cudaMemcpyHostToDevice);
    check_launch("copy to gpu");
    double elapsed_time_h2d = stop_timer(&timer);
    std::cout << "Host -> Device transfer bandwidth " << num_bytes / (elapsed_time_h2d / 1000.) / 1E9 << std::endl << std::endl;
  }
  
  // generate reference output
  {
    event_pair timer;
    start_timer(&timer);
    host_shift(text, text_host, shift_amount);
    double elapsed_time_host = stop_timer(&timer);
    std::cout << "Host (reference) solution bandwidth GB/sec: " << 2 * num_bytes / (elapsed_time_host / 1000.) / 1E9 << std::endl << std::endl;
  }

 //CUDA block size
  const int block_size = 256;

  std::cout << std::setw(45) << "Device Bandwidth GB/sec" << std::endl;
  std::cout << std::setw(70) << std::setfill('-') << " " << std::endl << std::setfill(' ');
  std::cout << std::setw(15) << " " << std::setw(15) << "char" << std::setw(15) << "uint" << std::setw(15) << "uint2" << std::endl;
  std::cout << std::setw(15) << "Problem Size MB" << std::endl;

  //loop through all the problem sizes and generate timing / bandwidth information for each
  //and also check correctness
  for (int i = 0; i < sizes_to_test.size(); ++i) {
    // generate GPU char output
    double elapsed_time_char = doGPUShiftChar(device_input_array, device_output_array, shift_amount, sizes_to_test[i], block_size);
    checkResults(text_host, device_output_array, sizes_to_test[i], "char");

    //make sure we don't falsely say the next kernel is correct because we've left the correct answer sitting in memory
    cudaMemset(device_output_array, 0, sizes_to_test[i]); 

    // generate GPU uint output
    double elapsed_time_uint = doGPUShiftUInt(device_input_array, device_output_array, shift_amount, sizes_to_test[i], block_size);
    checkResults(text_host, device_output_array, sizes_to_test[i], "uint");

    //make sure we don't falsely say the next kernel is correct because we've left the correct answer sitting in memory
    cudaMemset(device_output_array, 0, sizes_to_test[i]);

    //generate GPU uint2 output
    double elapsed_time_uint2 = doGPUShiftUInt2(device_input_array, device_output_array, shift_amount, sizes_to_test[i], block_size);
    checkResults(text_host, device_output_array, sizes_to_test[i], "uint2");

    //make sure we don't falsely say the next kernel is correct because we've left the correct answer sitting in memory
    cudaMemset(device_output_array, 0, sizes_to_test[i]);

    std::cout << std::setw(15) << sizes_to_test[i] / 1E6 << " " << 
                 std::setw(15) << 2 * sizes_to_test[i] / (elapsed_time_char / 1000.) / 1E9 << 
                 std::setw(15) << 2 * sizes_to_test[i] / (elapsed_time_uint / 1000.) / 1E9 << 
                 std::setw(15) << 2 * sizes_to_test[i] / (elapsed_time_uint2 / 1000.) / 1E9 << std::endl;
  }

  // deallocate memory
  cudaFree(device_input_array);
  cudaFree(device_output_array);
}
