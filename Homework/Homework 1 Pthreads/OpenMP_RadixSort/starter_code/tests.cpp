#include <algorithm>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"
#include "tests.h"
#include <sstream>

#define SIZE_MASK_TEST 8
#define STARTBIT_TEST 0 

std::vector<uint> computeBlockHistograms(std::vector<uint>& keys, uint numBlocks, uint numBuckets, uint numBits, uint startBit, uint blockSize);

std::vector<uint> reduceLocalHistoToGlobal(std::vector<uint>& blockHistograms, uint numBlocks, uint numBuckets);

std::vector<uint> scanGlobalHisto(std::vector<uint>& globalHisto, uint numBuckets);

std::vector<uint> computeBlockExScanFromGlobalHisto(uint numBuckets, uint numBlocks, std::vector<uint>& globalHistoExScan, std::vector<uint>& blockHistograms);

void populateOutputFromBlockExScan(std::vector<uint>& blockExScan, uint numBlocks, uint numBuckets, uint startBit, uint numBits, uint blockSize, std::vector<uint>& keys, std::vector<uint> &sorted);

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

void Test1() { 
	std::vector<uint> input = ReadVectorFromFile("test_files/input");
	uint blockSize = input.size() / 8;
	uint numBlocks = (input.size() + blockSize - 1) / blockSize;
	uint numBuckets = 1 << SIZE_MASK_TEST;
	std::vector<uint> blockHistograms = computeBlockHistograms(input, numBlocks, numBuckets, SIZE_MASK_TEST, STARTBIT_TEST, blockSize);
	TestCorrectness(ReadVectorFromFile("test_files/blockhistograms"), blockHistograms);
	std::cout << "Test 1 passes" << std::endl;
} 

void Test2() { 
	std::vector<uint> blockHistograms = ReadVectorFromFile("test_files/blockhistograms");
	std::vector<uint> input = ReadVectorFromFile("test_files/input");
	uint blockSize = input.size() / 8;
	uint numBlocks = (input.size() + blockSize - 1) / blockSize;
	uint numBuckets = 1 << SIZE_MASK_TEST;
	std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms, numBlocks, numBuckets);
	TestCorrectness(ReadVectorFromFile("test_files/globalhisto"), globalHisto);
	std::cout << "Test 2 passes" << std::endl;
}

void Test3() { 
	std::vector<uint> globalHisto = ReadVectorFromFile("test_files/globalhisto");
	uint numBuckets = 1 << SIZE_MASK_TEST;
	std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);
	TestCorrectness(ReadVectorFromFile("test_files/globalhistoexscan"), globalHistoExScan);
	std::cout << "Test 3 passes" << std::endl;
}

void Test4() {
	std::vector<uint> input = ReadVectorFromFile("test_files/input");
    uint blockSize = input.size() / 8;
	uint numBlocks = (input.size() + blockSize - 1) / blockSize;
	uint numBuckets = 1 << SIZE_MASK_TEST;
	std::vector<uint> globalHistoExScan = ReadVectorFromFile("test_files/globalhistoexscan");
	std::vector<uint> blockHistograms = ReadVectorFromFile("test_files/blockhistograms");
	std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets, numBlocks, globalHistoExScan, blockHistograms);
	TestCorrectness(ReadVectorFromFile("test_files/blockexscan"), blockExScan);
	std::cout << "Test 4 passes" << std::endl;
}

void Test5() { 
	std::vector<uint> blockExScan = ReadVectorFromFile("test_files/blockexscan");
	std::vector<uint> input = ReadVectorFromFile("test_files/input");
    uint blockSize = input.size() / 8;
    uint numBlocks = (input.size() + blockSize - 1) / blockSize;
    uint numBuckets = 1 << SIZE_MASK_TEST;
	std::vector<uint> sorted(input.size());
	populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, STARTBIT_TEST, SIZE_MASK_TEST, blockSize, input, sorted);
	TestCorrectness(ReadVectorFromFile("test_files/sorted"), sorted);
	std::cout << "Test 5 passes" << std::endl;
}


