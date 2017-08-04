"""
Tests for xform_2/4 in scf_utils.py
"""

import SuperCoolFast as scf
import numpy as np
import os


def test_xform():
    """
    Check whether energy is unchanged before and after ao_ints
    are xformed.
    [NOTE] The original check uses A = S^(-1/2) as the xform
    matrix, which is a weak check since A is symmetric. In the
    current version we add a xform using C, which is not symmetric
    and thus a more severe check.
    """
    # parse parameter from input file
    ao_ints, scf_params, e_ZZ_repul = \
        scf.init(os.path.dirname(__file__) + '/test_xform.yml')
    H = ao_ints['T'] + ao_ints['V']
    g = ao_ints['g4']
    nbas = scf_params['nbas']
    nel = scf_params['nel_alpha']
    # w/o xform
    eps, C, D, F = scf.scf(ao_ints, scf_params)
    energy1 = np.sum((F + H) * D) + e_ZZ_repul
    # energy using MO basis
    H_mo = scf.xform_2(H, C)
    g_mo = scf.xform_4(g, C)
    D_mo = np.diag(np.array([1 if i < nel else 0 for i in range(nbas)]))
    J_mo = np.einsum("pqrs,rs->pq", g_mo, D_mo)
    K_mo = np.einsum("prqs,rs->pq", g_mo, D_mo)
    F_mo = H_mo + 2. * J_mo - K_mo
    energy3 = np.sum((F_mo + H_mo) * D_mo) + e_ZZ_repul
    # w/ xform (do the SCF in symmetrically orthogonalized basis set)
    A = ao_ints['A']
    ao_ints['T'] = scf.xform_2(ao_ints['T'], A)
    ao_ints['V'] = scf.xform_2(ao_ints['V'], A)
    ao_ints['g4'] = scf.xform_4(ao_ints['g4'], A)
    ao_ints['S'] = np.eye(A.shape[0])
    ao_ints['A'] = np.eye(A.shape[0])
    eps, C, D, F = scf.scf(ao_ints, scf_params)
    H = ao_ints['T'] + ao_ints['V']
    energy2 = np.sum((F + H) * D) + e_ZZ_repul

    print(energy1, energy2, energy3)

    # check
    assert(np.allclose(energy1, energy2) == True)
    assert(np.allclose(energy1, energy3) == True)
