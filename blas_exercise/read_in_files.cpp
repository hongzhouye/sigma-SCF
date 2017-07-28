#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include "include/read_in_files.hpp"

using namespace std;

vector<double> read_matrix_file(const string& filename, int nrow, int ncol)
{
    // inititalize the output matrix (1D)
    vector<double> out (nrow*ncol, 0);
    // open the  file
    ifstream input(filename);
    double number;
    // import everything from the input file
    for(int currentRow=0; currentRow < nrow; currentRow++)
    {
        for(int currentCol=0; currentCol < ncol; currentCol++)
        {
            input >> out[currentCol*nrow + currentRow];
        }
    }
    return out;
}

/*
int main()
{
    vector<double> H (49, 0);
    H = read_matrix_file("C.data", 7, 7);
    for(int row=0; row<7; row++)
    {
      for(int col=0; col<7; col++)
      {
        printf("%6.6f \t", H[col*7 + row]);
      }
      printf("\n");
    }
}
*/
