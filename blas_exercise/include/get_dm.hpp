/* ----------------------------------------------------
    Header file for get_dm.cpp
---------------------------------------------------- */

#ifndef GET_DM_HPP
#define GET_DM_HPP

#include <iostream>
#include <vector>

typedef std::vector<double> matrix;

matrix get_dm(const matrix& C, int nbas, int nocc);

#endif
