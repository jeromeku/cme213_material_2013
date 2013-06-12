#include <algorithm>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"
#include "tests.h"
#include <sstream>

void TestCorrectness(const std::vector<uint>& reference, const std::vector<uint>& toTest) { 
	assert(reference.size() == toTest.size());
	for (uint i = 0; i < reference.size(); ++i) { 
        assert(reference[i] == toTest[i]);
	}
}

void WriteVectorToFile(const std::string& filename, std::vector<uint>& v) { 
	std::ofstream outfile(filename.c_str());
	if (!outfile.is_open()) std::cerr << "Failed to open the file." << std::endl;
	for (uint i = 0; i < v.size(); ++i) {
		outfile << v[i] << std::endl;
	}
	outfile.close();
}

uint StringToUint(const std::string& line) {
	std::stringstream buffer;
	uint res;
	buffer << line;
	buffer >> res;
	return res;
}

std::vector<uint> ReadVectorFromFile(const std::string& filename) { 
	std::ifstream infile(filename.c_str());
	if (!infile) std::cerr << "Failed to load the file." << std::endl;
	std::vector<uint> res;
	std::string line;
	while (true) {
		getline(infile, line);
		if (infile.fail()) break;
		res.push_back(StringToUint(line));	
	}
	return res;	
}

