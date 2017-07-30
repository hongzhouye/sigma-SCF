"""
Tests for DF verison of get_JK
"""

import SuperCoolFast as scf
import numpy as np
import os


def test_df():
    """
    Compare psi4 energy (w/ DF) and our energy (w/ DF)
    """
    # out code
    ao_ints, scf_params, e_ZZ_repul = \
        scf.init(os.path.dirname(__file__) + '/test_df.yml')
    eps, C, D, F = scf.scf(ao_ints, scf_params)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F + H) * D) + e_ZZ_repul

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

    # check
    assert(np.allclose(energy, psi4_energy) == True)
