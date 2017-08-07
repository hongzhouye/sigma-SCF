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
    # Spin-restricted sigma-SCF
    eps, C, D, F = scf.rsscf(ao_ints, test_scf_param, e_nuc)
    e_rsscf = scf.get_SCF_energy(ao_ints, F, D, False) + e_nuc
    # Spin-unrestricted sigma-SCF (not break spin symmetry)
    epss, Cs, Ds, Fs = scf.usscf(ao_ints, test_scf_param, e_nuc)
    e_usscf = scf.get_SCF_energy(ao_ints, Fs, Ds, True) + e_nuc
    # Spin-unrestricted sigma-SCF (break spin symmetry)
    test_scf_param['homo_lumo_mix'] = 3
    epss, Cs, Ds, Fs = scf.usscf(ao_ints, test_scf_param, e_nuc)
    e_usscf2 = scf.get_SCF_energy(ao_ints, Fs, Ds, True) + e_nuc
    # check
    assert(np.allclose(e_rsscf, e_usscf) == True)
    assert(e_rsscf > e_usscf2)
