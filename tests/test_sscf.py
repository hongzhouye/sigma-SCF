import numpy as np
import SuperCoolFast as scf
import os


def test_sscf():
    """
    Using physicist's water to test RSSCF and USSCF
    """
    # get integrals and scf parameters
    ao_ints, test_scf_param, e_nuc = scf.init(\
        os.path.dirname(__file__) + "/test_sscf.yml")
    # Spin-restricted sigma-SCF, DIIS
    test_scf_param['unrestricted'] = False
    eps, C, D, F = scf.sscf(ao_ints, test_scf_param, e_nuc)
    e_rsscf_diis = scf.get_SCF_energy(ao_ints['H'], F, D, False) + e_nuc
    # Spin-restricted sigma-SCF, ODA
    test_scf_param['opt'] = "oda"
    eps, C, D, F = scf.sscf(ao_ints, test_scf_param, e_nuc)
    e_rsscf_oda = scf.get_SCF_energy(ao_ints['H'], F, D, False) + e_nuc
    # Spin-unrestricted sigma-SCF (not break spin symmetry), DIIS
    test_scf_param['unrestricted'] = True
    test_scf_param['opt'] = "diis"
    epss, Cs, Ds, Fs = scf.sscf(ao_ints, test_scf_param, e_nuc)
    e_usscf_diis = scf.get_SCF_energy(ao_ints['H'], Fs, Ds, True) + e_nuc
    # Spin-unrestricted sigma-SCF (not break spin symmetry), ODA
    test_scf_param['unrestricted'] = True
    test_scf_param['opt'] = "oda"
    epss, Cs, Ds, Fs = scf.sscf(ao_ints, test_scf_param, e_nuc)
    e_usscf_oda = scf.get_SCF_energy(ao_ints['H'], Fs, Ds, True) + e_nuc
    # Spin-unrestricted sigma-SCF (break spin symmetry), DIIS
    test_scf_param['homo_lumo_mix'] = 3
    test_scf_param['opt'] = "diis"
    epss, Cs, Ds, Fs = scf.sscf(ao_ints, test_scf_param, e_nuc)
    e_usscf_diis_bs = scf.get_SCF_energy(ao_ints['H'], Fs, Ds, True) + e_nuc
    # check
    assert(np.allclose(e_rsscf_diis, e_rsscf_oda) == True)
    assert(np.allclose(e_rsscf_diis, e_usscf_diis) == True)
    assert(np.allclose(e_rsscf_diis, e_usscf_oda) == True)
    assert(e_rsscf_diis > e_usscf_diis_bs)
