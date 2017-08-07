import numpy as np
import SuperCoolFast as scf
import os


def test_ortho_ao():
    """
    test for uhf using (H2O)^+
    """
    ao_ints, test_scf_param, e_ZZ_repulsion = scf.init(\
        os.path.dirname(__file__) + "/test_ortho_ao.yml")
    eps, C, D, F = scf.scf(ao_ints, test_scf_param, e_ZZ_repulsion)
    H = ao_ints['T'] + ao_ints['V']
    energy = scf.get_SCF_energy(ao_ints, F, D, True) + e_ZZ_repulsion

    # psi4 setup
    import psi4
    mol = psi4.geometry("""
    1 2
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mol.update_geometry()

    bas = psi4.core.BasisSet.build(mol, target=test_scf_param['basis'])
    mints = psi4.core.MintsHelper(bas)

    # DENSITY FITTED
    psi4.set_options({"scf_type": "pk", "reference": "uhf"})
    psi4_energy = psi4.energy("SCF/cc-pVDZ", molecule = mol)
    assert(np.allclose(energy, psi4_energy) == True)

if __name__ == "__main__":
    test_uhf()
