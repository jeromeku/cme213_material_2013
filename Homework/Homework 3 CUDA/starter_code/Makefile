COMMON=../common

heat: 2dHeat_driver.o simParams.o Grid.o
	g++ -o heat 2dHeat_driver.o simParams.o Grid.o -L /cm/shared/apps/cuda50/toolkit/current/lib64 -lcudart -Wall -Winline -Wextra -Wno-strict-aliasing

2dHeat_driver.o: 2dHeat_driver.cu $(COMMON)/mp1-util.h simParams.h Grid.h gpuStencil.cu BC.h
	nvcc -c 2dHeat_driver.cu -O3 -arch=sm_20 -Xcompiler -Wall,-Wextra,-Wno-strict-aliasing -I $(COMMON)

simParams.o: simParams.cpp simParams.h
	g++ -c simParams.cpp -O3 -Wall -Wextra -Wno-strict-aliasing

Grid.o: Grid.cu Grid.h
	nvcc -c Grid.cu -O3 -arch=sm_20 -Xcompiler -Wall,-Wextra,-Wno-strict-aliasing

clean:
	rm -f *.o heat *.txt
