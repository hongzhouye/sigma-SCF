"""
Tests for psi4wrapper.py
"""

import SuperCoolFast as scf
import os


def test_diis():
    """
    Physicist's water w/ aug-cc-pVDZ should fix in 12 iterations w/ diis.
    """
    ao_ints, scf_params, e_ZZ_repul = \
        scf.init(os.path.dirname(__file__) + '/test_diis.yml')
    eps, C, D, F = scf.scf(ao_ints, scf_params)
