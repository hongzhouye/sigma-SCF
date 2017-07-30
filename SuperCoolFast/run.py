"""
Take everything from init (e.g. integrals, packed parameters)
and do the calculations
"""

from scf import *


def run(ao_ints, scf_params):
    """
    """
    eps, C, D, F = scf(ao_ints, scf_params)
    H = ao_ints['T'] + ao_ints['V']
    energy = np.sum((F + H) * D) + e_ZZ_repul
    print("FINAL ENERGY: {}\n".format(energy))
