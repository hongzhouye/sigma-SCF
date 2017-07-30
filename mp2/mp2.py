"""
MP2
"""

import numpy as np


def get_denominator_mp2(eps, nocc, nvirt):
    """
    build denominator of mp2 given energy levels
    """
    # grep occ/virt energy levels
    e1 = eps[:nocc]
    e2 = e1.copy()
    e3 = eps[nocc:]
    e4 = e3.copy()

    # reshape
    e1 = e1.reshape(nocc, 1, 1, 1)
    e2 = e2.reshape(nocc, 1, 1)
    e3 = e3.reshape(nvirt, 1)

    # build the denominator tensor
    return 1. / (e1 + e2 - e3 - e4)
