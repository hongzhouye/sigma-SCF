"""
Making a wrapper for psi4 that retrieves integrals and the nuclear-nuclear repulsion.
"""

import numpy as np
import psi4


def init(scf_params):
    """
    Sets up integral variables. Function inputs are molecule string and basis string. Returns nothing.
    """
    basis = scf_params['basis']
    mol_geometry = scf_params['geometry']

    ao_ints = {'V':'','T':'','S':'','g':'','A':''}

    mol = psi4.geometry(mol_geometry)
    mol.update_geometry()
    bas = psi4.core.BasisSet.build(mol, target=basis)
    aux_bas = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=basis)
    mints = psi4.core.MintsHelper(bas)

    nbf = mints.nbf()
    if (nbf > scf_params['max_nbf']):
        raise Exception("More than %d basis functions!" % scf_params['max_nbf'])

    ao_ints['V'] = np.array(mints.ao_potential())
    ao_ints['T'] = np.array(mints.ao_kinetic())
    ao_ints['S'] = np.array(mints.ao_overlap())
    ao_ints['g'] = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    ao_ints['A'] = np.array(A)

    e_nuclear_repulsion = mol.nuclear_repulsion_energy()
    nel = 0
    for i in range(mol.natom()):
        nel += mol.Z(i)
    nel = int(nel)

    return ao_ints, e_nuclear_repulsion, nel, nbf
