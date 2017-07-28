#include "include/get_nel.hpp"
#include <lawrap/blas.h>

int get_nel(const matrix& D, const matrix& S) {
    int n = D.size();
    double nel = LAWrap::dot(n, D.data(), 1, S.data(), 1);
    return (int) round(nel);
}
