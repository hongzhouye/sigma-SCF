"""
Tests for psi4wrapper.py
"""

import psi4wrapper as p4w
import numpy as np


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

def test_psi4wrapper():
    """
    test for psi4wrapper
    """
    ao_ints, e_nuclear_repulsion, nel, nbf = p4w.init(test_scf_param)

    T = ao_ints['T']
    V = ao_ints['V']
    g = ao_ints['g']
    S = ao_ints['S']
    A = ao_ints['A']

    # e_nuclear_repulsion is a float number
    assert(type(e_nuclear_repulsion) == float)

    # nel and nbf are positive integer
    assert(type(nel) == int)
    assert(nel > 0)
    assert(type(nbf) == int)
    assert(nbf > 0)

    # dimension check
    assert(T.shape == (nbf, nbf))
    assert(V.shape == (nbf, nbf))
    if(not test_scf_param['is_fitted']):
        assert(g.shape == (nbf, nbf, nbf, nbf))
    assert(S.shape == (nbf, nbf))
    assert(A.shape == (nbf, nbf))

    # symmetry check
    assert(np.allclose(T.T, T) == True)
    assert(np.allclose(V.T, V) == True)
    assert(np.allclose(S.T, S) == True)
    assert(np.allclose(A.T, A) == True)

    # A == S^(-1/2)
    assert(np.allclose(A @ S @ A, np.eye(nbf)) == True)
