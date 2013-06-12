#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <fstream>
#include <iostream>
#include "strided_range_iterator.h"

//You will need to call this functors from
//thrust functions in the code
//do not create new ones
//this can be the same as in create_cipher.cu

struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char> {
  //TODO
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "No cipher text given!" << std::endl;
    return 1;
  }

  //First load the text 
  std::ifstream ifs(argv[1], std::ios::binary);

  if (!ifs.good()) {
    std::cerr << "Couldn't open book file!" << std::endl;
    return 1;
  }

  //load the file into text
  std::vector<unsigned char> text;

  ifs.seekg(0, std::ios::end); //seek to end of file
  int length = ifs.tellg();    //get distance from beginning
  ifs.seekg(0, std::ios::beg); //move back to beginning

  text.resize(length);
  ifs.read((char *)&text[0], length);

  ifs.close();

  //we assume the cipher text has been sanitized
  thrust::device_vector<unsigned char> text_clean = text;

  //now we need to crack vignere cipher
  //first we need to determine the key length
  //use the kappa index of coincidence
  int keyLength = 0;
  {
    bool found = false;
    int i = 4;
    while (!found) {
      int numMatches; // = ?  TODO

      double ioc = numMatches / ((double)(text_clean.size() - i) / 26.); 

      std::cout << "Period " << i << " ioc: " << ioc << std::endl;
      if (ioc > 1.6) {
        if (keyLength == 0) {
          keyLength = i;
          i = 2 * i - 1; //check double the period to make sure
        }
        else if (2 * keyLength == i)
          found = true;
        else {
          std::cout << "Unusual pattern in text!" << std::endl;
          exit(1);
        }
      }
      ++i; 
    }
  }

  std::cout << "keyLength: " << keyLength << std::endl;

  //once we know the key length, then we can do frequency analysis on each pos mod length
  //allowing us to easily break each cipher independently
  //you will find the strided_range_iterator useful
  //it is located in strided_range_iterator.h and an example
  //of how to use it is located in the that file
  thrust::device_vector<unsigned char> text_copy = text_clean;
  thrust::device_vector<int> dShifts(keyLength);
  typedef thrust::device_vector<unsigned char>::iterator Iterator;

  //TODO : fill up the dShifts vector with the correct shifts

  //take the shifts and transform cipher text back to plain texxt
  //TODO : transform the cipher text back to the plain text

  thrust::host_vector<unsigned char> h_plain_text = text_clean; 

  std::ofstream ofs("plain_text.txt", std::ios::binary);
  ofs.write((char *)&h_plain_text[0], h_plain_text.size());
  ofs.close();

  return 0;
}
