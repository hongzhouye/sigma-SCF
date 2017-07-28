/* ----------------------------------------------------
    Calculate SCF electric energy using formula
        E = Tr[(F + H) * D] / 2
---------------------------------------------------- */

#include "include/get_eel.hpp"
#include <lawrap/blas.h>

double get_eel(const matrix& H, const matrix& F, const matrix& D, int nbas)
{
	// FH = H + F

	matrix FH = H;
	LAWrap::axpy(nbas * nbas, 1.0, F.data(), 1, FH.data(), 1);

	// FHD_2 = FH * D * 0.5

	matrix FHD_2(nbas * nbas, 0.);
	LAWrap::gemm('N', 'N', nbas, nbas, nbas, 0.5, FH.data(), nbas,
		D.data(), nbas, 0., FHD_2.data(), nbas);

	// energy = Tr[FHD_2]
	double energy = 0.;
	for (int i = 0; i < nbas; i++)
		energy += FHD_2[i + i * nbas];

	return energy;
}
