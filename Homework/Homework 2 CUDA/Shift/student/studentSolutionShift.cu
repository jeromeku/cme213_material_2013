// Repeating from the tutorial, just in case you haven't looked at it.

// "kernels" or __global__ functions are the entry points to code that executes on the GPU
// The keyword __global__ indicates to the compiler that this function is a GPU entry point.
// __global__ functions must return void, and may only be called or "launched" from code that
// executes on the CPU.

// This kernel implements a per element shift
// by naively loading one byte and shifting it
__global__ void shift_char(unsigned char *input_array
                         , unsigned char *output_array
                         , unsigned char shift_amount
                         , unsigned int array_length)
{
  //TOOD: fill in
}

//Here we load 4 bytes at a time instead of just 1
//to improve the bandwidth due to a better memory
//access pattern
__global__ void shift_int(unsigned int *input_array
                        , unsigned int *output_array
                        , unsigned int shift_amount
                        , unsigned int array_length) 
{
  //TODO: fill in
}

//Here we go even further and load 8 bytes
//does it make a further improvement?
__global__ void shift_int2(uint2 *input_array
                         , uint2 *output_array
                         , unsigned int shift_amount
                         , unsigned int array_length) 
{
  //TODO: fill in
}

//the following three kernels launch their respective kernels
//and report the time it took for the kernel to run

double doGPUShiftChar(unsigned char *d_input
                    , unsigned char *d_output
                    , unsigned char shift_amount
                    , int text_size
                    , int block_size)
{
  //compute your grid dimensions
  event_pair timer;
  start_timer(&timer);
  // launch kernel
  check_launch("gpu shift cipher char");
  return stop_timer(&timer);
}

double doGPUShiftUInt(unsigned char *d_input
                    , unsigned char *d_output
                    , unsigned char shift_amount
                    , int text_size
                    , int block_size)
{
  //compute your grid dimensions
  //compute 4 byte shift value
  event_pair timer;
  start_timer(&timer);
  // launch kernel
  check_launch("gpu shift cipher uint");
  return stop_timer(&timer);
}

double doGPUShiftUInt2(unsigned char *d_input
                     , unsigned char *d_output
                     , unsigned char shift_amount
                     , int text_size
                     , int block_size)
{
  //compute your grid dimensions
  //compute 4 byte shift value

  event_pair timer;
  start_timer(&timer);
  // launch kernel
  check_launch("gpu shift cipher uint2");
  return stop_timer(&timer);
}
