"""
Tests for psi4wrapper.py
"""

import SuperCoolFast as scf
import numpy as np
import os


def test_diis():
    """
    We pick the physicist's water molecule in a relatively large
    basis set (aug-cc-pVDZ), whose uhf and rhf solutions are the
    same. And we test that energies of the following four cases
    are identical:
        RHF + DIIS
        RHF + ODA
        UHF + DIIS
        UHF + ODA
    Note for DIIS, we in addition require the convergence is achieved
    in 30 iterations.
    """
    # get integrals from psi4
    ao_ints, scf_params, e_ZZ_repul = \
        scf.init(os.path.dirname(__file__) + '/test_opt.yml')
    # RHF, DIIS
    scf_params['max_iter'] = 30
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    E_rhf_diis = scf.get_SCF_energy(ao_ints, F, D, False) + e_ZZ_repul
    # RHF, ODA
    scf_params['max_iter'] = 300
    scf_params['opt'] = "oda"
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    E_rhf_oda = scf.get_SCF_energy(ao_ints, F, D, False) + e_ZZ_repul
    # UHF, DIIS
    scf_params['max_iter'] = 30
    scf_params['opt'] = "diis"
    scf_params['unrestricted'] = True
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    E_uhf_diis = scf.get_SCF_energy(ao_ints, F, D, True) + e_ZZ_repul
    # UHF, ODA
    scf_params['max_iter'] = 300
    scf_params['opt'] = "oda"
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    E_uhf_oda = scf.get_SCF_energy(ao_ints, F, D, True) + e_ZZ_repul
    # RHF, psi4
    import psi4
    mol = psi4.geometry(scf_params['geometry'])
    mol.update_geometry()
    ctrl_string = "SCF/" + scf_params['basis']
    psi4.set_options({"scf_type": "pk"})
    psi4_E_rhf = psi4.energy(ctrl_string, molecule = mol)
    # check
    assert(np.allclose(E_rhf_diis, E_rhf_oda) == True)
    assert(np.allclose(E_rhf_diis, E_uhf_diis) == True)
    assert(np.allclose(E_rhf_diis, E_uhf_oda) == True)
    assert(np.allclose(E_rhf_diis, psi4_E_rhf) == True)
