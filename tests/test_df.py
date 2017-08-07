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
    # get integrals
    ao_ints, scf_params, e_ZZ_repul = \
        scf.init(os.path.dirname(__file__) + '/test_df.yml')
    # rhf
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    energy_rhf = scf.get_SCF_energy(ao_ints['H'], F, D, False) + e_ZZ_repul

    # uhf
    scf_params['unrestricted'] = True
    scf_params['homo_lumo_mix'] = 0
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    energy_uhf = scf.get_SCF_energy(ao_ints['H'], F, D, True) + e_ZZ_repul

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
    assert(np.allclose(energy_rhf, psi4_energy) == True)
    assert(np.allclose(energy_uhf, psi4_energy) == True)
