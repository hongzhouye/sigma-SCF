/* ----------------------------------------------------
    Header file for get_eel.cpp
---------------------------------------------------- */

#ifndef GET_EEL_HPP
#define GET_EEL_HPP

#include <iostream>
#include <vector>

typedef std::vector<double> matrix;

double get_eel(const matrix& H, const matrix& F, const matrix& D, int nbas);

#endif
