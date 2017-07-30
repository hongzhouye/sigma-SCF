"""
MP2 module
"""

import numpy as np


def get_denominator_mp2(eps, nocc, nvirt):
    """
    build denominator of mp2 given energy levels
    """
    # grep occ/virt energy levels
    e1 = eps[:nocc]
    e2 = e1.copy()
    e3 = eps[nocc:]
    e4 = e3.copy()

    # reshape
    e1 = e1.reshape(nocc, 1, 1, 1)
    e2 = e2.reshape(nocc, 1, 1)
    e3 = e3.reshape(nvirt, 1)

    # build the denominator tensor
    return 1. / (e1 + e2 - e3 - e4)


def eri_ao_to_mo_mp2(C, g, nocc):
    """
    Do the basis transformation from AO to MO for eri tensor (g)
    """
    C_occ = C[:, :nocc]
    C_virt = C[:, nocc:]
    g = np.einsum('ab, acde->bcde', C_virt, g)
    g = np.einsum('ab, cdae->cdbe', C_virt, g)
    g = np.einsum('ab, cade->cbde', C_occ, g)
    g = np.einsum('ab, cdea->cdeb', C_occ, g)
    return g


def get_mp2_energy(eps, C, g, nocc):
    """
    Get MP2 energy.
    """
    nbas = C.shape[0]
    denom = get_denominator_mp2(eps, nocc, nbas - nocc)
    g = eri_ao_to_mo_mp2(C, g, nocc)
    e_os = np.einsum("iajb, iajb, abij->", g, g, denom)
    g_ss = g - np.swapaxes(g, 1, 3)
    e_ss = np.einsum("iajb, iajb, abij->", g_ss, g, denom)
    return e_ss / 3. + 1.2 * e_os
