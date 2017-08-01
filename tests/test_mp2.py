import numpy as np
import SuperCoolFast as scf
import mp2
import os


def test_mp2():
    """
    test for energies
    """
    ao_ints, test_scf_param, e_ZZ_repulsion = scf.init(\
        os.path.dirname(__file__) + "/test_energies.yml")
    eps, C, D, F = scf.scf(ao_ints, test_scf_param)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F+H)*D) + e_ZZ_repulsion

    # psi4 setup
    import psi4
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mol.update_geometry()

    bas = psi4.core.BasisSet.build(mol, target="cc-pVDZ")
    mints = psi4.core.MintsHelper(bas)

    # DF-MP2
    psi4.set_options({"scf_type": "df"})
    psi4.set_options({"MP2_type": "CONV"})
    psi4_energy = psi4.energy("MP2/cc-pVDZ", molecule = mol)
    psi4_energy_MP2 = psi4.get_variable('SCS-MP2 TOTAL ENERGY')

    test_scf_param.update({"method": "MP2"})
    test_scf_param.update({"is_fitted": "True"})
    eps, C, D, F = scf.scf(ao_ints, test_scf_param)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F+H)*D) + e_ZZ_repulsion
    energy_corr = mp2.get_mp2_energy(eps, C, ao_ints['g4'], test_scf_param['nel'])
    assert(np.allclose(energy + energy_corr, psi4_energy_MP2) == True)

