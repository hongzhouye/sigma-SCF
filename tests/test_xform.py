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
    H = ao_ints['H']
    g = ao_ints['g4']
    nbas = scf_params['nbas']
    nel = scf_params['nel_alpha']
    # w/o xform
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    energy1 = scf.get_SCF_energy(ao_ints['H'], F, D, False) + e_ZZ_repul
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
    ao_ints['H'] = scf.xform_2(ao_ints['H'], A)
    ao_ints['g4'] = scf.xform_4(ao_ints['g4'], A)
    ao_ints['S'] = np.eye(A.shape[0])
    ao_ints['A'] = np.eye(A.shape[0])
    eps, C, D, F = scf.scf(ao_ints, scf_params, e_ZZ_repul)
    energy2 = scf.get_SCF_energy(ao_ints['H'], F, D, False) + e_ZZ_repul

    print(energy1, energy2, energy3)

    # check
    assert(np.allclose(energy1, energy2) == True)
    assert(np.allclose(energy1, energy3) == True)
