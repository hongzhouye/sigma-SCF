"""
Tests for xform_2/4 in scf_utils.py
"""

import SuperCoolFast as scf
import numpy as np
import os


def test_xform():
    """
    check whether energy is unchanged before and after ao_ints
    are xformed.
    """
    # w/o xform
    ao_ints, scf_params, e_ZZ_repul = \
        scf.init(os.path.dirname(__file__) + '/test_xform.yml')
    eps, C, D, F = scf.scf(ao_ints, scf_params)
    H = ao_ints['T'] + ao_ints['V']
    energy1 = np.sum((F + H) * D) + e_ZZ_repul
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

    # check
    assert(np.allclose(energy1, energy2) == True)
