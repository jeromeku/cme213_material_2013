/* File: grid.cpp
 * This file contains the classes Grid and Field that are used to represent 
 * the physical grid on which we will solve the heat PDE. 
 * It also contains the methods that will communicate the boundaries 
 * between successive iterations. 
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

class Field { // this class allows us to use the operator() easily
    private:
        std::vector<double> *gridPtr; // pointer to the physical grid
        int gx; // width of the grid. Used to compute the 1D indices from a pair of coordinates
    public:
        int offset; // offset of the field in the physical grid (the grid contains both curr and prev)
        double operator()(int xpos, int ypos) const { 
            return (*gridPtr)[offset + ypos * gx + xpos];
        }
        double& operator()(int xpos, int ypos) { 
            return (*gridPtr)[offset + ypos * gx + xpos];
        }
        Field() : gridPtr(NULL), gx(0), offset(0) {} 
        Field(std::vector<double> *ptr, int gxP, int initialOffset) : gridPtr(ptr), gx(gxP), offset(initialOffset) {}
        void setMembers(std::vector<double> *ptr, int gxP, int initialOffset) { gridPtr = ptr; offset = initialOffset;  gx = gxP; }
};

class Grid {
    public:
        Field curr, prev; // two states of the grid
        Grid(const simParams &params, bool debug);
        ~Grid() { }
        int gx() const {return gx_;}
        int gy() const {return gy_;}
        int nx() const {return nx_;}
        int ny() const {return ny_;}
        int borderSize() const {return borderSize_;}
        int rank() const {return ourRank_;}
        void swapState() {int temp = prev.offset; prev.offset = curr.offset; curr.offset = temp;}

        void transferHaloDataBlocking(); // blocking communication
        void transferHaloDataNonBlocking(); // non-blocking communication
        void waitForSends(); //block until sends are finished
        void waitForRecvs(); //block until receives are finished
        
        void initializeIC(const simParams &params, int ourRank, int totalNumProcessors); // initializes the grid
        void updateBCs(const simParams &params); // update the boundary conditions between two iterations. Takes 
                                                 // care of the cases where the top and bottom boundaries need to be 
                                                 // updated

        void saveStateToFile(std::string identifier) const;
        friend std::ostream & operator<<(std::ostream &os, const Grid& grid);

    private:
        std::vector<double> grid_;
        int gx_, gy_;               //total grid extents
        int nx_, ny_;               //non-boundary region
        int borderSize_;            //number of halo cells
        int procTop_;               //MPI processor number
        int procBot_;               //of our neighbors
                                    //negative if not used
        
        int ourRank_;
        bool debug_;

        std::vector<MPI_Request> send_requests_;
        std::vector<MPI_Request> recv_requests_;

        //prevent copying and assignment since they are not implemented
        //and don't make sense for this class
        Grid(const Grid &);
        Grid& operator=(const Grid &);

};

std::ostream& operator<<(std::ostream& os, const Grid &grid) {
    os << std::setprecision(15) << std::fixed;
    for (int y = grid.gy() - 1; y != -1; --y) {
        for (int x = 0; x < grid.gx(); x++) {
            os << std::setw(5) << grid.curr(x, y) << " ";
        }
        os << std::endl;
    }
    os << std::endl;
    return os;
}

void Grid::initializeIC(const simParams &params, int ourRank, int totalNumProcessors) {
    const double dx = params.dx();
    const double dy = params.dy();  

    int offsetY = ourRank * ny_; // the number of cells below us (processors of lower rank)
    int mod = params.ny() % totalNumProcessors; 
    if (ourRank >= mod) { 
        offsetY += mod; // the processors at the beginning have a bigger ny_
    }

    for (int i = 0; i < gx_; ++i) {
        for (int j = 0; j < gy_; ++j) { 
           (this -> prev)(i, j) = sin(i * dx) * sin((j + offsetY) * dy);
        }
    }

}

Grid::Grid(const simParams &params, bool debug) {
    debug_ = debug;


    //need to figure out which processor we are and who our neighbors are...
    MPI_SAFE_CALL( MPI_Comm_rank(MPI_COMM_WORLD, &ourRank_) );

    int totalNumProcessors;
    MPI_SAFE_CALL( MPI_Comm_size(MPI_COMM_WORLD, &totalNumProcessors) );

    //based on total number of processors and grid configuration
    //determine our neighbors
    procTop_ = -1;
    procBot_ = -1;

    //1D decomposition - horizontal stripes
    if (ourRank_ > 0) {
        procBot_ = ourRank_ - 1;
    }
    if (ourRank_ < totalNumProcessors - 1) {
        procTop_ = ourRank_ + 1;
    }
    //figure out dimensions and how big we need to allocate
    nx_ = params.nx();

    ny_ = params.ny() / totalNumProcessors;
    if (ourRank_ < params.ny() % totalNumProcessors) // to deal with cases where there is a remainder
        ++ny_;                                       // in the division of ny by the number of procs

    if (params.order() == 2) 
        borderSize_ = 1;
    else if (params.order() == 4)
        borderSize_ = 2;
    else if (params.order() == 8)
        borderSize_ = 4;

    assert(nx_ > 2 * borderSize_);
    assert(ny_ > 2 * borderSize_);

    gx_ = nx_ + 2 * borderSize_;
    gy_ = ny_ + 2 * borderSize_;
   
    if (debug) { 
        printf("%d: (%d, %d) (%d, %d) top: %d bot: %d\n", \
                ourRank_, nx_, ny_, gx_, gy_, procTop_, procBot_);
    }

    grid_.resize(2 * gx_ * gy_);
    
    prev.setMembers(&grid_, gx_, 0);
    curr.setMembers(&grid_, gx_, gx_ * gy_);

    initializeIC(params, ourRank_, totalNumProcessors);
    //create the copy of the grid we need for ping-ponging
    std::copy(grid_.begin(), grid_.begin() + gx_ * gy_, grid_.begin() + gx_ * gy_);
}

void Grid::waitForSends() {
    // TODO
}

void Grid::waitForRecvs() {
    // TODO
}

// This function sends the data between processors in a 
// non blocking way.
void Grid::transferHaloDataNonBlocking() {
    // TODO
}

void Grid::transferHaloDataBlocking() {
    // TODO
}

void Grid::updateBCs(const simParams &params) {
    double alpha = params.alpha();
    double dt = params.dt();
    // all processors need to update left and right boundaries
    for (int j = 0; j < gy_; ++j)  { 
        for (int i = 0; i < borderSize_; ++i) {
            (this -> curr)(i, j) = (this -> prev)(i, j) * exp(-2 * alpha * dt); 
            (this -> curr)(nx_ + borderSize_ + i, j) = (this -> prev)(nx_ + borderSize_ + i, j) * exp(-2 * alpha * dt); 
        }
    }

    // if no neighbor up, we need to update the BCs there
    if (procTop_ < 0) {
        for (int j = ny_ + borderSize_; j < gy_; ++j) { 
            for (int i = 0; i < gx_; ++i) { 
                (this -> curr)(i, j) = (this -> prev)(i, j) * exp(-2 * alpha * dt);    
            }
        }
    }

    // if no neighbor down, we need to update the BCs there
    if (procBot_ < 0) {
        for (int j = 0; j < borderSize_; ++j) { 
            for (int i = 0; i < gx_; ++i) { 
                (this -> curr)(i, j) = (this -> prev)(i, j) * exp(-2 * alpha * dt); 
            }
        }
    }
}

void Grid::saveStateToFile(std::string identifier) const {
    std::stringstream ss;
    ss << "grid" << ourRank_ << "_" << identifier << ".txt";
    std::ofstream ofs(ss.str().c_str());
    
    ofs << *this << std::endl;

    ofs.close();
}


