import numpy as np
import SuperCoolFast as scf
import os

test_scf_param = {
    'nel': "",
    'nbas': "",
    'conv': 5,
    'opt': "None",
    'max_nbf': 120,
    'guess': "Core",
    'max_iter': 100,
    'charge': 0,
    'basis': 'sto-3g',
    'is_fitted': True,
    'geometry': '''
    O
    H   1   1.1
    H   1   1.1 2   104
    '''
}

def test_energies():
    """
    test for energies
    """
    ao_ints, test_scf_param, e_ZZ_repulsion = scf.init(\
        os.path.dirname(__file__) + "/test_energies.yml")

    eps, C, D, F = scf.scf(ao_ints, test_scf_param)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F+H)*D) + e_ZZ_repulsion

    # psi4
    import psi4
    mol = psi4.geometry("""
    O
    H 1 1.1
    H 1 1.1 2 104
    """)
    mol.update_geometry()

    bas = psi4.core.BasisSet.build(mol, target="cc-pVDZ")
    mints = psi4.core.MintsHelper(bas)

    psi4.set_options({"scf_type": "df"})
    psi4_energy = psi4.energy("SCF/cc-pVDZ", molecule = mol)

    assert(np.allclose(energy, psi4_energy) == True)
