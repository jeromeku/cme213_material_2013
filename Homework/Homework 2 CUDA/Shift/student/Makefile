COMMON=../../common

shift: shift_driver.cu studentSolutionShift.cu $(COMMON)/mp1-util.h
	nvcc -o shift shift_driver.cu -O3 -arch=sm_20 -I $(COMMON)

clean:
	rm -f shift
