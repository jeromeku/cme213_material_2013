#ifndef GRID_H__
#define GRID_H__

#include <vector>
#include <ostream>
#include <string>

class Grid {
  public:
    Grid(int gx, int gy, float ic);
    Grid(const Grid &);
    ~Grid();

    int gx() const {return gx_;}
    int gy() const {return gy_;}

    void saveStateToFile(const std::string &identifier);

    void toGPU();
    void fromGPU();

    friend std::ostream & operator<<(std::ostream &os, const Grid& grid);

    void swap(Grid &a, Grid &b);

    std::vector<float> hGrid_;
    float *            dGrid_;

  private:
    int gx_, gy_;             //total grid extents
};

#endif
