#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

//You will need to call these functors from
//thrust functions in the code
//do not create new ones

//returns true if the char is not a lowercase letter
struct isnot_lowercase_alpha : thrust::unary_function<unsigned char, bool>{
  //TODO
};

//convert an uppercase letter into a lowercase one
//do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char>{
  //TODO
};

//apply a shift with appropriate wrapping
struct apply_shift : thrust::binary_function<unsigned char, int, unsigned char> {
  //TODO
};

//Print the top 5 letter frequencies
//Print the top 10 bigram frequencies
void printLetterBigramFrequency(const thrust::device_vector<unsigned char> &text)
{
  //WARNING: make sure you handle the case of not all letters
  //and not all bigrams appearing in the text.  It is very likely
  //that not all 26 * 26 bigrams will appear in actual english text.
  //We may also test your code with samples like 'aaabbbbddd' that even have
  //less than 5 distinct letters.  Make sure your code doesn't crash :)

  //first calculate letter frequency
  {
    //TODO
  }

  //calculate bigram frequencies
  //see warning above
  {
    //TODO
  }
}

int main(int argc, char **argv) {
    if (argc != 3) {
      std::cerr << "Didn't supply plain text and period!" << std::endl;
      return 1;
    }

    std::ifstream ifs(argv[1], std::ios::binary);
    if (!ifs.good()) {
        std::cerr << "Couldn't open text file!" << std::endl;
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

    thrust::device_vector<unsigned char> text_clean;
    int numElements; //the number of characters in the cleaned text
    //sanitize input to contain only a-z lowercase
    //TODO : put the result in text_clean, make sure to resize text_clean to the correct size!

    std::cout << "Before ciphering!" << std::endl << std::endl;
    printLetterBigramFrequency(text_clean);

    unsigned int period = atoi(argv[2]);

    thrust::device_vector<unsigned int> shifts(period);
    //TODO fill in shifts using thrust random number generation

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    //place the cipher text in device_cipher_text
    //TODO

    std::cout << "After ciphering!" << std::endl << std::endl;
    printLetterBigramFrequency(device_cipher_text);

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;
    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char *)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
