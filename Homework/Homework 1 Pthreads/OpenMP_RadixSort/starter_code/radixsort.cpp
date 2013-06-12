#include <algorithm>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"
#include "tests.h"

#define SIZE_TEST_VECTOR 400000000
#define SIZE_MASK 16 // must be a divider of 32 for this program to work correctly

typedef unsigned int uint;

/* Function: computeBlockHistograms
 * --------------------------------
 * Splits keys into numBlocks and computes an histogram with numBuckets buckets 
 * Remember that numBuckets and numBits are related; same for blockSize and numBlocks.
 * Should work in parallel.
 */
std::vector<uint> computeBlockHistograms(std::vector<uint>& keys, uint numBlocks, uint numBuckets, uint numBits, uint startBit, uint blockSize) {
    std::vector<uint> blockHistograms(numBlocks * numBuckets, 0);
    // TO DO
    return blockHistograms;
}

/* Function: reduceLocalHistoToGlobal
 * ----------------------------------
 * Takes as input the local histogram of size numBuckets * numBlocks and "merges"
 * them into a global histogram of size numBuckets.
 */ 
std::vector<uint> reduceLocalHistoToGlobal(std::vector<uint>& blockHistograms, uint numBlocks, uint numBuckets) {
	std::vector<uint> globalHisto(numBuckets, 0);
	// TO DO
    return globalHisto;
}

/* Function: computeBlockExScanFromGlobalHisto 
 * -------------------------------------------
 * Takes as input the globalHistoExScan that contains the global histogram after the scan
 * and the local histogram in blockHistograms. Returns a local histogram that will be used
 * to populate the sorted array.
 */
std::vector<uint> computeBlockExScanFromGlobalHisto(uint numBuckets, uint numBlocks, std::vector<uint>& globalHistoExScan, std::vector<uint>& blockHistograms) { 
    std::vector<uint> blockExScan(numBuckets * numBlocks, 0);
	// TO DO 
    return blockExScan;
}

/* Function: populateOutputFromBlockExScan
 * ---------------------------------------
 * Takes as input the blockExScan produced by the splitting of the global histogram
 * into blocks and populates the vector sorted.
 */
void populateOutputFromBlockExScan(std::vector<uint>& blockExScan, uint numBlocks, uint numBuckets, uint startBit, uint numBits, uint blockSize, std::vector<uint>& keys, std::vector<uint> &sorted) {
    // TO DO
}

/* Function: scanGlobalHisto
 * -------------------------
 * This function should simply scan the global histogram.
 */
std::vector<uint> scanGlobalHisto(std::vector<uint>& globalHisto, uint numBuckets) {
	std::vector<uint> globalHistoExScan(numBuckets, 0);
    // TO DO  
	return globalHistoExScan;
}

/* Function: radixSortParallelPass
 * -------------------------------
 * A pass of radixSort on numBits starting after startBit.
 */
void radixSortParallelPass(std::vector<uint> &keys, std::vector<uint> &sorted, uint numBits, uint startBit, uint blockSize)
{
    uint numBuckets = 1 << numBits;
    uint numBlocks = (keys.size() + blockSize - 1) / blockSize;
    
	//go over each block and compute its local histogram
    std::vector<uint> blockHistograms = computeBlockHistograms(keys, numBlocks, numBuckets, numBits, startBit, blockSize);
	
    //first reduce all the local histograms into a global one
    std::vector<uint> globalHisto = reduceLocalHistoToGlobal(blockHistograms, numBlocks, numBuckets);

    //now we scan this global histogram
    std::vector<uint> globalHistoExScan = scanGlobalHisto(globalHisto, numBuckets);
	
    //now we do a local histogram in each block and add in the 
    //global value to get global position
    std::vector<uint> blockExScan = computeBlockExScanFromGlobalHisto(numBuckets, numBlocks, globalHistoExScan, blockHistograms); 
	
	//populate the sorted vector
	populateOutputFromBlockExScan(blockExScan, numBlocks, numBuckets, startBit, numBits, blockSize, keys, sorted);
}

int radixSortParallel(std::vector<uint> &keys, std::vector<uint> &keys_tmp, uint numBits) {
    for (uint startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortParallelPass(keys, keys_tmp, numBits, startBit, keys.size()/8); 
        radixSortParallelPass(keys_tmp, keys, numBits, startBit + numBits, keys.size()/4); 
    }
	return 0;
}

void radixSortSerialPass(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint startBit, uint numBits)
{
    uint numBuckets = 1 << numBits;
    uint mask = numBuckets - 1;

    //compute the frequency histogram
    std::vector<uint> histogramRadixFrequency(numBuckets);
    for (uint i = 0; i < keys.size(); ++i) {
        uint key = (keys[i] >> startBit) & mask;
        ++histogramRadixFrequency[key];
    }

    //now scan it
    std::vector<uint> exScanHisto(numBuckets, 0);
    for (uint i = 1; i < numBuckets; ++i) {
        exScanHisto[i] = exScanHisto[i-1] + histogramRadixFrequency[i-1];
        histogramRadixFrequency[i-1] = 0;
    }

    histogramRadixFrequency[numBuckets - 1] = 0;

    //now add the local to the global and scatter the result
    for (uint i = 0; i < keys.size(); ++i) {
        uint key = (keys[i] >> startBit) & mask;

        uint localOffset = histogramRadixFrequency[key]++;
        uint globalOffset = exScanHisto[key] + localOffset;

        keys_radix[globalOffset] = keys[i];
    }
}

int radixSortSerial(std::vector<uint> &keys, std::vector<uint> &keys_radix, uint numBits) {
    assert(numBits <= 16);
    for (int startBit = 0; startBit < 32; startBit += 2 * numBits) {
        radixSortSerialPass(keys,       keys_radix, startBit  ,  numBits);
        radixSortSerialPass(keys_radix, keys,       startBit+numBits, numBits);
    }
	return 0;
}

void initializeRandomly(std::vector<uint>& keys) { 
    for (uint i = 0; i < keys.size(); ++i) {
        keys[i] = rand();
    }
}

int main(void)
{
	// Test1();
	// Test2();
	// Test3();
	// Test4();
	// Test5();
    
    
    /* 
    // this is the timing test and the final correctness test. Uncomment when you are done

	std::vector<uint> keys(SIZE_TEST_VECTOR);
    std::vector<uint> keys_stl = keys; // need to copy because keys is going to be modified	
	
	
	// stl sort
    double startstl = omp_get_wtime();
    std::sort(keys_stl.begin(), keys_stl.end());
    double endstl = omp_get_wtime();
 
	// serial radix sort
    std::vector<uint> keys_serial = keys; // need to copy because keys is going to be modified
    std::vector<uint> bufferVector(keys.size());
    double startRadixSerial = omp_get_wtime();
    radixSortSerial(keys_serial, bufferVector, SIZE_MASK);
    double endRadixSerial = omp_get_wtime();
    TestCorrectness(keys_stl, keys_serial); // verify that the array is well sorted

	// parallel radix sort
    double startRadixParallel = omp_get_wtime();
    radixSortParallel(keys, bufferVector, SIZE_MASK);
    double endRadixParallel = omp_get_wtime();
    TestCorrectness(keys_stl, keys);
  	std::cout << "Parallel Radix Sort is correct" << std::endl;

    std::cout << "stl: " << endstl - startstl << std::endl;
    std::cout << "serial radix: " << endRadixSerial - startRadixSerial << std::endl;
    std::cout << "parallel radix: " << endRadixParallel - startRadixParallel << std::endl;
    */
    return 0;
}
