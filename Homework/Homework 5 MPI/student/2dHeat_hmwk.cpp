/* Machine Problem 5
 * 2D Heat Diffusion w/MPI
 * 
 * You will implement the familiar 2D Heat Diffusion from the
 * previous homework on CPUs with MPI.
 * 
 * You have been given the simParams class (simparams.cpp) updated
 * with all necessary parameters and the outline of
 * Grid class (grid.cpp) that you will fill in.  You are also given the 
 * stencil calculations since you have already implemented
 * them in the previous homework.
 *
 * You are also given a macro - MPI_SAFE_CALL which you should
 * wrap all MPI calls with to always check error return codes.
 *
 * You will implement two communications schemes: blocking and non-blocking.
 */


#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdlib.h>

#include "mpi.h"
#include "simparams.cpp"
#include "grid.cpp"

inline double stencil2(const Grid &grid, int x, int y, double xcfl, double ycfl) {
    return grid.prev(x, y) + 
           xcfl * (grid.prev(x+1, y) + grid.prev(x-1, y) - 2 * grid.prev(x, y)) + 
           ycfl * (grid.prev(x, y+1) + grid.prev(x, y-1) - 2 * grid.prev(x, y));
}

inline double stencil4(const Grid &grid, int x, int y, double xcfl, double ycfl) {
    return grid.prev(x, y) + 
           xcfl * (   -grid.prev(x+2, y) + 16 * grid.prev(x+1, y) -
                    30 * grid.prev(x, y) + 16 * grid.prev(x-1, y) - grid.prev(x-2, y)) + 
           ycfl * (   -grid.prev(x, y+2) + 16 * grid.prev(x, y+1) -
                    30 * grid.prev(x, y) + 16 * grid.prev(x, y-1) - grid.prev(x, y-2));
}

inline double stencil8(const Grid &grid, int x, int y, double xcfl, double ycfl) {
    return grid.prev(x, y) +
           xcfl*(-9*grid.prev(x+4,y) + 128*grid.prev(x+3,y) - 1008*grid.prev(x+2,y) + 8064*grid.prev(x+1,y) -
                                                  14350*grid.prev(x, y) + 
                 8064*grid.prev(x-1,y) - 1008*grid.prev(x-2,y) + 128*grid.prev(x-3,y) - 9*grid.prev(x-4,y)) + 
           ycfl*(-9*grid.prev(x,y+4) + 128*grid.prev(x,y+3) - 1008*grid.prev(x,y+2) + 8064*grid.prev(x,y+1) -
                                                  14350*grid.prev(x,y) +
                8064*grid.prev(x,y-1) -1008*grid.prev(x,y-2) + 128*grid.prev(x,y-3) - 9*grid.prev(x,y-4));
}


template<bool blocking> void computation(Grid &grid, const simParams &params) {    
    double xcfl = params.xcfl();
    double ycfl = params.ycfl();

    for (int i = 0; i < params.iters(); ++i) {
        // Process the inner region
        // TODO
        
        // Update the Boundary conditions
        grid.updateBCs(params);
        grid.swapState();
        
        // Send and receive the appropriate boundaries
        // TODO
    }
    grid.swapState();

}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Please supply a parameter file!" << std::endl;
        exit(1);
    }

    MPI_Init(&argc, &argv);

    simParams params(argv[1], true);
    Grid grid(params, true);

    grid.saveStateToFile("init"); //save our initial state, useful for making sure we
                                  //got setup and BCs right

    double start = MPI_Wtime();

    if (params.blocking()) {
        computation<true>(grid, params);
    }
    else {
        computation<false>(grid, params);
    }

    double end = MPI_Wtime();

    if (grid.rank() == 0) {
        std::cout << params.iters() << " iterations on a " << params.nx() << " by " 
                  << params.ny() << " grid took: " << end - start << " seconds." << std::endl;
    }
    grid.saveStateToFile("final"); //final output for correctness checking of computation

    MPI_Finalize(); 
    return 0;
}
