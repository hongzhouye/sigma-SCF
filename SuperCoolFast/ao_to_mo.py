import numpy as np

def eri_ao_to_mo_MP2(C, eri, n_occ_orbs, n_virt_orbs):
    C_occ = C[:, 0:n_occ_orbs]
    C_virt = C[:, n_occ_orbs:n_virt_orbs]
    eri = np.einsum('ab, acde->bcde', C_virt, eri)
    eri = np.einsum('ab, cdae->cdbe', C_virt, eri)
    eri = np.einsum('ab, cade->cbde', C_occ, eri)
    eri = np.einsum('ab, cdae->cdbe', C_occ, eri)
    return eri
