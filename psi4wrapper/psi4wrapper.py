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
    is_fitted = scf_params['is_fitted']

    ao_ints = {'V':'','T':'','S':'','g':'','A':''}

    mol = psi4.geometry(mol_geometry)
    mol.update_geometry()
    bas = psi4.core.BasisSet.build(mol, target=basis)
    mints = psi4.core.MintsHelper(bas)

    if(is_fitted):
        # build the 3-index ERIs for density fitting of JK
        aux_bas = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=basis)
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        mints_index_3 = mints.ao_eri(zero_bas, aux_bas, bas, bas)
        mints_index_3 = np.squeeze(mints_index_3)
        # build JK metric and invert
        metric_JK = mints.ao_eri(zero_bas, aux_bas, zero_bas, aux_bas)
        metric_JK.power(-0.5, 1.e-14)
        metric_JK = np.squeeze(metric_JK)
        # construct (P|(lambda)(sigma))
        g_3 = np.einsum('pq, qls->pls', metric_JK, mints_index_3)
        ao_ints['g3'] = g_3

    nbf = mints.nbf()
    if (nbf > scf_params['max_nbf']):
        raise Exception("More than %d basis functions!" % scf_params['max_nbf'])

    ao_ints['V'] = np.array(mints.ao_potential())
    ao_ints['T'] = np.array(mints.ao_kinetic())
    ao_ints['S'] = np.array(mints.ao_overlap())
    ao_ints['g4'] = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    ao_ints['A'] = np.array(A)

    e_nuclear_repulsion = mol.nuclear_repulsion_energy()
    nel = 0
    for i in range(mol.natom()):
        nel += mol.Z(i)
    nel = int(nel)

    return ao_ints, e_nuclear_repulsion, nel, nbf
