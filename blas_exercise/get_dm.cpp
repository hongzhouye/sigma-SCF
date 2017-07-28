/* ----------------------------------------------
    Calculate density matrix using formula
        D = C[:, :nocc] * C[:. :nocc].T
---------------------------------------------- */

#include <lawrap/blas.h>
#include "include/get_dm.hpp"


matrix get_dm(const matrix& C, int nbas, int nocc)
{
    matrix D(nbas * nbas, 0.);
    LAWrap::gemm('N', 'T', nbas, nbas, nocc, 2., C.data(), nbas,
        C.data(), nbas, 0., D.data(), nbas);

    return D;
}
