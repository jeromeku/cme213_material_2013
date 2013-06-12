#include <algorithm>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "omp.h"
#include <fstream>
#include <string> 

typedef unsigned int uint;

#ifndef TESTS_H
#define TESTS_H

void TestCorrectness(const std::vector<uint>& reference, const std::vector<uint>& toTest);

void WriteVectorToFile(const std::string& filename, std::vector<uint>& v);

std::vector<uint> ReadVectorFromFile(const std::string& filename); 

uint StringToUint(const std::string& line);

void Test1();

void Test2();

void Test3();

void Test4();

void Test5();

#endif
