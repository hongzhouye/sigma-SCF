import numpy as np
import SuperCoolFast as scf
import os


def test_rhf():
    """
    test for rhf
    """
    ao_ints, test_scf_param, e_ZZ_repulsion = scf.init(\
        os.path.dirname(__file__) + "/test_rhf.yml")
    eps, C, D, F = scf.scf(ao_ints, test_scf_param, e_ZZ_repulsion)
    energy = scf.get_SCF_energy(ao_ints['H'], F, D, False) + e_ZZ_repulsion

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

    # DENSITY FITTED
    psi4.set_options({"scf_type": "df"})
    psi4_energy = psi4.energy("SCF/cc-pVDZ", molecule = mol)
    assert(np.allclose(energy, psi4_energy) == True)
