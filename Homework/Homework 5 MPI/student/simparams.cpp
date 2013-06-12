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

#ifndef _simparams
#define _simparams

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

class simParams {
    public:
        simParams(const char *filename, bool verbose); //parse command line
                                                       //does no error checking
        simParams(); //use some default values

        int    nx()         const {return nx_;}
        int    ny()         const {return ny_;}
        double lx()         const {return lx_;}
        double ly()         const {return ly_;}
        double alpha()      const {return alpha_;}
        int    iters()      const {return iters_;}
        double dx()         const {return dx_;}
        double dy()         const {return dy_;}
        double dt()         const {return dt_;}
        int    order()      const {return order_;}
        double xcfl()       const {return xcfl_;}
        double ycfl()       const {return ycfl_;}
        bool   blocking()       const {return blocking_;}

    private:
        int    nx_, ny_;     //number of grid points in each dimension
        double lx_, ly_;     //extent of physical domain in each dimension
        double alpha_;       //thermal conductivity
        double dt_;          //timestep
        int    iters_;       //number of iterations to do
        double dx_, dy_;     //size of grid cell in each dimension
        double xcfl_, ycfl_; //cfl numbers in each dimension
        int    order_;       //order of discretization
        bool  blocking_;     //blocking or non-blocking communication scheme

        void calcDtCFL();
};

simParams::simParams() {
    nx_ = ny_ = 10;
    lx_ = ly_ = 1;
    alpha_ = 1;
    iters_ = 1000;
    order_ = 2;
    int gx = nx_ + order_;
    int gy = ny_ + order_;
    dx_ = lx_ / (gx - 1);
    dy_ = ly_ / (gy - 1);
    blocking_ = true;

    calcDtCFL();
}

simParams::simParams(const char *filename, bool verbose) {
    std::ifstream ifs(filename);

    if (!ifs.good()) {
        std::cerr << "Couldn't open parameter file!" << std::endl;
        exit(1);
    }

    ifs >> nx_ >> ny_;
    ifs >> lx_ >> ly_;
    ifs >> alpha_;
    ifs >> iters_;
    ifs >> order_;
    ifs >> blocking_;

    ifs.close();
    
    int gx = nx_ + order_;
    int gy = ny_ + order_;

    dx_ = lx_ / (gx - 1);
    dy_ = ly_ / (gy - 1);

    calcDtCFL();

    int rank;

    MPI_SAFE_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    if (verbose && rank == 0) {
        printf("nx: %d ny: %d\nlx %f: ly: %f\nalpha: %f\niterations: %d\norder: %d\nblocking: %d\n", 
                nx_, ny_, lx_, ly_, alpha_, iters_, order_, blocking_);
        printf("dx: %f dy: %f\ndt: %f xcfl: %f ycfl: %f\n", 
                dx_, dy_, dt_, xcfl_, ycfl_);
    }
}

void simParams::calcDtCFL() {
    //check cfl number and make sure it is ok
    if (order_ == 2) {
        //make sure we come in just under the limit
        dt_ = (.5 - .001) * (dx_ * dx_ * dy_ * dy_) / (alpha_ * (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (alpha_ * dt_) / (dx_ * dx_);
        ycfl_ = (alpha_ * dt_) / (dy_ * dy_);
    }
    else if (order_ == 4) {
        dt_ = (.5 - .001) * (12 * dx_ * dx_ * dy_ * dy_) / (16 * alpha_ * (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (alpha_ * dt_) / (12 * dx_ * dx_);
        ycfl_ = (alpha_ * dt_) / (12 * dy_ * dy_);
    }
    else if (order_ == 8) {
        dt_ = (.5 - .01) * (5040 * dx_ * dx_ * dy_ * dy_) / (8064 * alpha_ * (dx_ * dx_ + dy_ * dy_));
        xcfl_ = (alpha_ * dt_) / (5040 * dx_ * dx_);
        ycfl_ = (alpha_ * dt_) / (5040 * dy_ * dy_);
    }
    else {
        std::cerr << "Unsupported discretization order. Exiting..." << std::endl;
        exit(1);
    }
}

#endif
