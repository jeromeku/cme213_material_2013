COMMON=../../common

pagerank: pagerank_driver.cu studentSolutionPagerank.cu $(COMMON)/mp1-util.h
	nvcc -o pagerank pagerank_driver.cu -O3 -arch=sm_20 -I $(COMMON)

clean:
	rm -f pagerank
