"""
Making a wrapper for psi4 that retrieves integrals and the nuclear-nuclear repulsion.
"""

import numpy as np
import psi4

V = None
g = None
T = None
S = None
A = None
e_nuclear_repulsion = None

def init(mol_geometry, basis):
    """
    Sets up integral variables. Function inputs are molecule string and basis string. Returns nothing.
    """
    mol = psi4.geometry(mol_geometry)
    mol.update_geometry()
    bas = psi4.core.BasisSet.build(mol, target=basis)
    mints = psi4.core.MintsHelper(bas)
	
    nbf = mints.nbf()
    if (nbf > 100):
        raise Exception("More than 100 basis functions!")

    V = np.array(mints.ao_potential())
    T = np.array(mints.ao_kinetic())
    S = np.array(mints.ao_overlap())
    g = np.array(mints.ao_eri())

    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)
    
    e_nuclear_repulsion = mol.nuclear_repulsion_energy()

    return
 
